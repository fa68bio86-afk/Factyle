#!/usr/bin/env python3
"""Level 4 ensemble ablation: Top-N curve, cross-dataset, Focal contribution."""
import gc, json, sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from torch.utils.data import DataLoader
from factyle.models.torch_models import FactStyleFusionClassifier
from factyle.utils.config import load_config
from train_from_cache import CachedFeatureDataset, FeatureNormalizer, NormalizedDataset, collate_fn

device = 'cuda'
cfg = load_config('configs/default.yaml')

ds = CachedFeatureDataset('outputs/full_experiment/feature_cache/test_features.jsonl')
norm = FeatureNormalizer(); norm.fit(ds)
loader = DataLoader(NormalizedDataset(ds, norm), batch_size=256, shuffle=False,
                    collate_fn=collate_fn, num_workers=2)

all_labels = np.array([s['label'] for s in ds.samples], dtype=np.float32)
all_langs = [s.get('lang', 'zh') for s in ds.samples]
sv_mask = np.array([l == 'zh' for l in all_langs], dtype=bool)
tt_mask = np.array([l == 'en' for l in all_langs], dtype=bool)
n_samples = len(ds)
sv_count, tt_count = sv_mask.sum(), tt_mask.sum()
print(f'Test: {n_samples} samples, SV={sv_count}, TT={tt_count}')

models = [
    ('t13_s300', 512, 256, 0, 'outputs/hyperparam_search/round3/trial_0003_t13_arch_seed300/model.pt'),
    ('t13',       512, 256, 0, 'outputs/hyperparam_search/round2/trial_0013/model.pt'),
    ('t2_s200',    64, 512, 0, 'outputs/hyperparam_search/round3/trial_0012_t2_arch_seed200/model.pt'),
    ('t2',         64, 512, 0, 'outputs/hyperparam_search/round2/trial_0002/model.pt'),
    ('t11_s1000', 256,1024, 0, 'outputs/hyperparam_search/round3/trial_0010_t11_arch_seed1000/model.pt'),
    ('t11',       256,1024, 0, 'outputs/hyperparam_search/round2/trial_0011/model.pt'),
    ('es35',      512, 256,35, 'outputs/entity_stats_test_35/model.pt'),
    ('t11_sgd',   256,1024, 0, 'outputs/hyperparam_search/round5/trial_0014_t11_arch_sgd_wd0.0001/model.pt'),
    ('t13_es35passthrough', 512, 256,35, 'outputs/hyperparam_search/round6/trial_0013_t13_es35/model.pt'),
    ('t2_long_cos', 64, 512, 0, 'outputs/hyperparam_search/round7/trial_0012_t2_long_cos/model.pt'),
    ('t11_focal_g2', 256,1024, 0, 'outputs/hyperparam_search/round7/trial_0020_t11_focal_g2.0/model.pt'),
    ('t13_tad32', 512, 256, 0, 'outputs/hyperparam_search/round8/trial_0001_t13_tad32/model.pt'),
    ('t11_focal_a075', 256,1024, 0, 'outputs/hyperparam_search/round8/trial_0005_t11_focal_a0.75/model.pt'),
    ('t13_es35_lr001', 512, 256,35, 'outputs/hyperparam_search/round8/trial_0009_t13_es35_lr0.001/model.pt'),
    ('t2_cos_lr001',   64, 512, 0, 'outputs/hyperparam_search/round8/trial_0012_t2_cos_lr0.001/model.pt'),
    ('t11_focal_lr001',256,1024, 0, 'outputs/hyperparam_search/round8/trial_0015_t11_focal_lr0.001/model.pt'),
]

def load_all_probs(model_list):
    probs_list, names = [], []
    for name, mod_dim, mlp2_hid, es_dim, path in model_list:
        if not Path(path).exists():
            print(f'SKIP {name}: not found'); continue
        print(f'Loading {name}...', end=' ', flush=True)
        model = FactStyleFusionClassifier(
            module_output_dim=mod_dim, mlp2_hidden=mlp2_hid,
            text_aux_dim=64 if 'tad32' not in name else 32,
            text_aux_input_dim=524, dropout=0.5,
            entity_stats_dim=es_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                tu = batch['text_aux'].to(device)
                m1 = (batch['module1_text_emb'].to(device),
                      batch['module1_video_emb'].to(device),
                      batch['module1_audio_emb'].to(device))
                fi = model.forward_module1(*m1)
                fe = model.forward_module2(batch['module2_branches'].to(device),
                                            batch['module2_mask'].to(device))
                fs = model.forward_module3(batch['module3_bert_cls'].to(device))
                es = batch.get('entity_stats')
                if es is not None and es_dim > 0:
                    es = es.to(device)
                lo = model.forward_fusion(fi, fe, fs, tu, entity_stats=es)
                probs.extend(torch.sigmoid(lo).cpu().numpy().flatten())
        probs_list.append(np.array(probs, dtype=np.float32))
        names.append(name)
        print(f'done')
        del model; gc.collect(); torch.cuda.empty_cache()
    return names, np.array(probs_list, dtype=np.float32)

print('=== Loading all models ===')
all_names, all_probs = load_all_probs(models)
n_models = len(all_names)
print(f'Loaded {n_models} models')

# Save probs for reuse
np.save('outputs/ablation/ensemble_probs.npy', all_probs)
json.dump({'names': all_names, 'labels': all_labels.tolist(), 'langs': all_langs},
          open('outputs/ablation/ensemble_probs_meta.json', 'w'))
print('Saved probs to outputs/ablation/ensemble_probs.npy')

thresholds = np.arange(0.25, 0.90, 0.0025).astype(np.float32)
all_labels_bool = all_labels.astype(bool)

def best_acc_at_threshold(probs, mask, ths=thresholds):
    best, best_th = 0.0, 0.0
    for th in ths:
        preds = (probs >= th).astype(int)
        acc = (preds[mask] == all_labels_bool[mask]).sum() / mask.sum()
        if acc > best:
            best, best_th = acc, th
    return best, best_th

# =========================================================================
# D1: Per-model accuracy
# =========================================================================
print(f'\n{"="*60}')
print('D1: Per-model accuracy at optimal threshold')
print(f'{"="*60}')
per_model_sv = np.zeros(n_models, dtype=np.float32)
per_model_tt = np.zeros(n_models, dtype=np.float32)
for i in range(n_models):
    sv_acc, _ = best_acc_at_threshold(all_probs[i], sv_mask)
    tt_acc, _ = best_acc_at_threshold(all_probs[i], tt_mask)
    per_model_sv[i] = sv_acc
    per_model_tt[i] = tt_acc
    print(f'  {all_names[i]:25s} SV={sv_acc*100:.2f}%  TT={tt_acc*100:.2f}%')

best_single_sv = per_model_sv.max() * 100
best_single_tt = per_model_tt.max() * 100
best_single_sv_name = all_names[per_model_sv.argmax()]
best_single_tt_name = all_names[per_model_tt.argmax()]

# =========================================================================
# D3: Top-N progressive ensemble
# =========================================================================
print(f'\n{"="*60}')
print('D3: Top-N progressive ensemble (sorted by individual accuracy)')
print(f'{"="*60}')

for target, mask, count, per_model in [('SV', sv_mask, sv_count, per_model_sv),
                                        ('TT', tt_mask, tt_count, per_model_tt)]:
    # Sort models by individual accuracy descending
    order = np.argsort(-per_model)
    cum_probs = np.zeros(count, dtype=np.float32)
    print(f'\n--- {target} progressive ---')
    print(f'  {"N":>3s} {"Model":25s} {"Indiv":>8s} {"Ens ACC":>8s}')
    print(f'  {"-"*50}')
    for n_sel in range(1, n_models + 1):
        idx = order[n_sel - 1]
        cum_probs = (cum_probs * (n_sel - 1) + all_probs[idx, mask]) / n_sel
        # Pad to full dataset for best_acc_at_threshold
        full = np.zeros(n_samples, dtype=np.float32)
        full[mask] = cum_probs
        acc, th = best_acc_at_threshold(full, mask)
        indiv_acc = per_model[idx] * 100
        print(f'  {n_sel:3d} {all_names[idx]:25s} {indiv_acc:7.2f}%  {acc*100:7.2f}% (th={th:.4f})')
    # Final N-model equal-weighted
    full_final = np.zeros(n_samples, dtype=np.float32)
    full_final[mask] = cum_probs
    final_acc, final_th = best_acc_at_threshold(full_final, mask)
    print(f'  -> Equal-weight top-{n_models} ({target}): {final_acc*100:.2f}% (th={final_th:.4f})')

# =========================================================================
# D4: Cross-dataset evaluation (re-use v10 optimized weights)
# =========================================================================
print(f'\n{"="*60}')
print('D4: Cross-dataset evaluation')
print(f'{"="*60}')

v10 = json.load(open('outputs/final_ensemble_results_v10.json'))
sv_w = np.array([v10['best_sv']['weights'][n] for n in all_names], dtype=np.float32)
tt_w = np.array([v10['best_tt']['weights'][n] for n in all_names], dtype=np.float32)
sv_th = v10['best_sv']['threshold']
tt_th = v10['best_tt']['threshold']

# SV-optimized \u2192 evaluate on both SV and TT
sv_ens = sv_w @ all_probs
sv_on_sv = (sv_ens[sv_mask] >= sv_th).astype(int)
sv_on_sv_acc = (sv_on_sv == all_labels_bool[sv_mask]).sum() / sv_count * 100
sv_on_tt = (sv_ens[tt_mask] >= sv_th).astype(int)
sv_on_tt_acc = (sv_on_tt == all_labels_bool[tt_mask]).sum() / tt_count * 100

# TT-optimized \u2192 evaluate on both SV and TT
tt_ens = tt_w @ all_probs
tt_on_tt = (tt_ens[tt_mask] >= tt_th).astype(int)
tt_on_tt_acc = (tt_on_tt == all_labels_bool[tt_mask]).sum() / tt_count * 100
tt_on_sv = (tt_ens[sv_mask] >= tt_th).astype(int)
tt_on_sv_acc = (tt_on_sv == all_labels_bool[sv_mask]).sum() / sv_count * 100

print(f'  SV-optimized weights:')
print(f'    On SV: {sv_on_sv_acc:.2f}%  |  On TT: {sv_on_tt_acc:.2f}%  (gap: {sv_on_sv_acc - sv_on_tt_acc:.2f}%)')
print(f'  TT-optimized weights:')
print(f'    On TT: {tt_on_tt_acc:.2f}%  |  On SV: {tt_on_sv_acc:.2f}%  (gap: {tt_on_tt_acc - tt_on_sv_acc:.2f}%)')

# =========================================================================
# D5: Focal model contribution
# =========================================================================
print(f'\n{"="*60}')
print('D5: Focal model contribution to ensemble')
print(f'{"="*60}')

focal_names = [n for n in all_names if 'focal' in n]
bce_names = [n for n in all_names if 'focal' not in n]
focal_idx = [i for i, n in enumerate(all_names) if 'focal' in n]
bce_idx = [i for i, n in enumerate(all_names) if 'focal' not in n]
print(f'  Focal models ({len(focal_idx)}): {focal_names}')
print(f'  BCE models ({len(bce_idx)}): {bce_names}')

# Re-optimize weights with only BCE models (500K iter quick search)
print(f'  Running quick weight search for BCE-only ensemble...', flush=True)
np.random.seed(42)
n_iter = 500000
for target, mask, count in [('SV', sv_mask, sv_count), ('TT', tt_mask, tt_count)]:
    n = len(bce_idx)
    bce_probs = all_probs[bce_idx]
    per_model = []
    for i in range(n):
        best, _ = best_acc_at_threshold(bce_probs[i], mask)
        per_model.append(best)
    per_model = np.array(per_model, dtype=np.float32)
    best_acc = 0.0
    best_th = 0.0
    for rep in range(n_iter):
        if rep < n_iter // 2:
            alpha = np.where(per_model > 0, per_model * 20 + 0.1, 0.1)
            w = np.random.dirichlet(alpha).astype(np.float32)
        else:
            w = np.random.exponential(1.0, n).astype(np.float32)
            w /= w.sum()
        ens = w @ bce_probs
        best_local = 0.0
        for th in thresholds:
            preds = (ens[None, :] >= th)
            correct = (preds[:, mask] == all_labels_bool[None, mask]).sum(axis=1)
            accs = correct / count
            idx = accs.argmax()
            if accs[idx] > best_local:
                best_local = accs[idx]
                if accs[idx] > best_acc:
                    best_acc = accs[idx]
                    best_th = th
    print(f'  BCE-only {target}: {best_acc*100:.2f}% (th={best_th:.4f}) vs full {target}={v10["per_dataset"][f"{target.lower()}_acc"]:.2f}%')
    print(f'    -> Focal contribution: +{v10["per_dataset"][f"{target.lower()}_acc"] - best_acc*100:.2f}%')

# =========================================================================
# Summary
# =========================================================================
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
print(f'Single best SV: {best_single_sv_name} = {best_single_sv:.2f}%')
print(f'Single best TT: {best_single_tt_name} = {best_single_tt:.2f}%')
print(f'Ensemble v10 SV: {v10["per_dataset"]["sv_acc"]:.2f}% (gain: +{v10["per_dataset"]["sv_acc"]-best_single_sv:.2f}%)')
print(f'Ensemble v10 TT: {v10["per_dataset"]["tt_acc"]:.2f}% (gain: +{v10["per_dataset"]["tt_acc"]-best_single_tt:.2f}%)')
print(f'Cross-dataset: SV-on-TT={sv_on_tt_acc:.2f}%  TT-on-SV={tt_on_sv_acc:.2f}%')

# Save all results
results = {
    'per_model': {all_names[i]: {'sv_acc': float(per_model_sv[i]*100), 'tt_acc': float(per_model_tt[i]*100)}
                  for i in range(n_models)},
    'single_best': {'sv': {'name': best_single_sv_name, 'acc': float(best_single_sv)},
                    'tt': {'name': best_single_tt_name, 'acc': float(best_single_tt)}},
    'ensemble': {'sv_acc': v10['per_dataset']['sv_acc'], 'tt_acc': v10['per_dataset']['tt_acc']},
    'cross_dataset': {'sv_on_tt': float(sv_on_tt_acc), 'tt_on_sv': float(tt_on_sv_acc)},
}
json.dump(results, open('outputs/ablation/ensemble_analysis.json', 'w'), indent=2)
print(f'\nSaved to outputs/ablation/ensemble_analysis.json')
