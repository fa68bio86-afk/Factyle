#!/usr/bin/env python3
"""Load v10 ensemble weights and compute per-class Precision/Recall/F1 for both datasets."""
import gc, json, sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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

# Same model list as v10
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

print('Loading models...')
probs_list = []
names = []
for name, mod_dim, mlp2_hid, es_dim, path in models:
    if not Path(path).exists():
        print(f'SKIP {name}: not found'); continue
    print(f'  {name}...', end=' ', flush=True)
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
    print('done')
    del model; gc.collect(); torch.cuda.empty_cache()

all_probs = np.array(probs_list, dtype=np.float32)

# Load v10 weights
result = json.load(open('outputs/final_ensemble_results_v10.json'))
sv_w = np.array([result['best_sv']['weights'][n] for n in names], dtype=np.float32)
tt_w = np.array([result['best_tt']['weights'][n] for n in names], dtype=np.float32)
sv_th = result['best_sv']['threshold']
tt_th = result['best_tt']['threshold']

# Compute ensemble predictions
sv_ens = sv_w @ all_probs
tt_ens = tt_w @ all_probs
sv_preds = (sv_ens >= sv_th).astype(int)
tt_preds = (tt_ens >= tt_th).astype(int)

def compute_metrics(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    fake_p = precision_score(y_true, y_pred, pos_label=1) * 100
    fake_r = recall_score(y_true, y_pred, pos_label=1) * 100
    fake_f = f1_score(y_true, y_pred, pos_label=1) * 100
    real_p = precision_score(y_true, y_pred, pos_label=0) * 100
    real_r = recall_score(y_true, y_pred, pos_label=0) * 100
    real_f = f1_score(y_true, y_pred, pos_label=0) * 100
    return {
        'accuracy': round(acc, 2),
        'macro_f1': round(macro_f1, 2),
        'fake_precision': round(fake_p, 2),
        'fake_recall': round(fake_r, 2),
        'fake_f1': round(fake_f, 2),
        'real_precision': round(real_p, 2),
        'real_recall': round(real_r, 2),
        'real_f1': round(real_f, 2),
    }

sv_metrics = compute_metrics(all_labels[sv_mask].astype(int), sv_preds[sv_mask], 'SV')
tt_metrics = compute_metrics(all_labels[tt_mask].astype(int), tt_preds[tt_mask], 'TT')

print('\n' + '='*70)
print(f'{"Metric":<20} {"FakeSV":>10} {"FakeTT":>10}')
print('='*70)
print(f'{"Accuracy":<20} {sv_metrics["accuracy"]:>9.2f}% {tt_metrics["accuracy"]:>9.2f}%')
print(f'{"Macro-F1":<20} {sv_metrics["macro_f1"]:>9.2f}% {tt_metrics["macro_f1"]:>9.2f}%')
print(f'{"Fake-Precision":<20} {sv_metrics["fake_precision"]:>9.2f}% {tt_metrics["fake_precision"]:>9.2f}%')
print(f'{"Fake-Recall":<20} {sv_metrics["fake_recall"]:>9.2f}% {tt_metrics["fake_recall"]:>9.2f}%')
print(f'{"Fake-F1":<20} {sv_metrics["fake_f1"]:>9.2f}% {tt_metrics["fake_f1"]:>9.2f}%')
print(f'{"Real-Precision":<20} {sv_metrics["real_precision"]:>9.2f}% {tt_metrics["real_precision"]:>9.2f}%')
print(f'{"Real-Recall":<20} {sv_metrics["real_recall"]:>9.2f}% {tt_metrics["real_recall"]:>9.2f}%')
print(f'{"Real-F1":<20} {sv_metrics["real_f1"]:>9.2f}% {tt_metrics["real_f1"]:>9.2f}%')

# Save detailed metrics
detailed = {
    'sv': sv_metrics,
    'tt': tt_metrics,
}
with open('outputs/final_ensemble_v10_detailed_metrics.json', 'w') as f:
    json.dump(detailed, f, indent=2)
print(f'\nDetailed metrics saved to outputs/final_ensemble_v10_detailed_metrics.json')
