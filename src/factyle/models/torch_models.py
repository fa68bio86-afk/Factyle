"""Optional PyTorch modules for the full neural version."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

from typing import Optional, Tuple


if TORCH_AVAILABLE:

    class CoAttentionBlock(nn.Module):
        """Symmetric bidirectional multi-head co-attention.

        Given two sequences (left, right), computes attention in both directions
        (left->right and right->left) and fuses via residual + LayerNorm.
        Implements x3.2.3 bidirectional cross-modal attention.
        """

        def __init__(self, dim: int, num_heads: int = 4) -> None:
            super().__init__()
            assert dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.w_q = nn.Linear(dim, dim)
            self.w_k = nn.Linear(dim, dim)
            self.w_v = nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            B = left.size(0)
            l = left.unsqueeze(1)
            r = right.unsqueeze(1)

            def _attend(q, k, v):
                B_ = q.size(0)
                q = q.view(B_, 1, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(B_, 1, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(B_, 1, self.num_heads, self.head_dim).transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = F.softmax(attn, dim=-1)
                out = attn @ v
                out = out.transpose(1, 2).contiguous().view(B_, -1)
                return out

            out_l2r = _attend(self.w_q(l), self.w_k(r), self.w_v(r))
            out_r2l = _attend(self.w_q(r), self.w_k(l), self.w_v(l))
            out = (out_l2r + out_r2l) / 2.0
            out = self.proj(out)
            return self.norm(left + out)

    class GatedTriModalConsistency(nn.Module):
        """Bidirectional cross-modal co-attention with modality gap.

        ARCHITECTURE.md x3.2.3

        Uses bidirectional co-attention between modality pairs to infer each
        modality from the other two, then computes modality gaps.
        Learnable Missing Embeddings (x3.2.4) replace unavailable modalities.

        Input: text_emb, video_emb, audio_emb  (batch, 1024)
        Output: FINT_FACT (batch, 256)
        """

        def __init__(
            self,
            imagebind_dim: int = 1024,
            proj_dim: int = 256,
            hidden_dim: int = 256,
            num_heads: int = 4,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.proj_t = nn.Linear(imagebind_dim, proj_dim)
            self.proj_v = nn.Linear(imagebind_dim, proj_dim)
            self.proj_a = nn.Linear(imagebind_dim, proj_dim)
            self.co_attn_va = CoAttentionBlock(proj_dim, num_heads)
            self.co_attn_at = CoAttentionBlock(proj_dim, num_heads)
            self.co_attn_vt = CoAttentionBlock(proj_dim, num_heads)
            self.mlp = nn.Sequential(
                nn.Linear(proj_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.missing_text = nn.Parameter(torch.zeros(imagebind_dim))
            self.missing_video = nn.Parameter(torch.zeros(imagebind_dim))
            self.missing_audio = nn.Parameter(torch.zeros(imagebind_dim))
            self._init_weights()

        def _init_weights(self) -> None:
            for p in self.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                elif p.dim() == 1 and p.size(0) == self.missing_text.size(0):
                    nn.init.normal_(p, std=0.02)

        def forward(
            self, text_emb: torch.Tensor, video_emb: torch.Tensor, audio_emb: torch.Tensor
        ) -> torch.Tensor:
            bsz = text_emb.size(0)

            # All three modalities missing: return zeros
            if (
                text_emb.abs().sum() == 0
                and video_emb.abs().sum() == 0
                and audio_emb.abs().sum() == 0
            ):
                return torch.zeros(bsz, self.mlp[-1].out_features, device=text_emb.device)

            # Replace missing modalities with learned embeddings
            if text_emb.abs().sum() == 0:
                text_emb = self.missing_text.unsqueeze(0).expand(bsz, -1)
            if video_emb.abs().sum() == 0:
                video_emb = self.missing_video.unsqueeze(0).expand(bsz, -1)
            if audio_emb.abs().sum() == 0:
                audio_emb = self.missing_audio.unsqueeze(0).expand(bsz, -1)

            t_h = self.proj_t(text_emb)
            v_h = self.proj_v(video_emb)
            a_h = self.proj_a(audio_emb)

            # Bidirectional co-attention between modality pairs
            c_va = self.co_attn_va(v_h, a_h)
            c_at = self.co_attn_at(a_h, t_h)
            c_vt = self.co_attn_vt(v_h, t_h)

            # Modality gaps: |projected - cross-modal prediction|
            gap_t = (t_h - c_va).abs()
            gap_v = (v_h - c_at).abs()
            gap_a = (a_h - c_vt).abs()

            gaps = torch.cat([gap_t, gap_v, gap_a], dim=-1)
            fint_fact = self.mlp(gaps)
            return fint_fact

    class SimpleTriModalConsistency(nn.Module):
        """C1: Simplest possible cross-modal fusion.

        Projects each modality (1024->256), averages the 3 projected vectors
        element-wise, then Linear(256,256)->ReLU. No co-attention, no modality gaps.
        Missing modality handling is preserved.
        """

        def __init__(
            self,
            imagebind_dim: int = 1024,
            proj_dim: int = 256,
            hidden_dim: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.proj_t = nn.Linear(imagebind_dim, proj_dim)
            self.proj_v = nn.Linear(imagebind_dim, proj_dim)
            self.proj_a = nn.Linear(imagebind_dim, proj_dim)
            self.mlp = nn.Sequential(
                nn.Linear(proj_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.missing_text = nn.Parameter(torch.zeros(imagebind_dim))
            self.missing_video = nn.Parameter(torch.zeros(imagebind_dim))
            self.missing_audio = nn.Parameter(torch.zeros(imagebind_dim))
            self._init_weights()

        def _init_weights(self) -> None:
            for p in self.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)

        def forward(
            self, text_emb: torch.Tensor, video_emb: torch.Tensor, audio_emb: torch.Tensor
        ) -> torch.Tensor:
            bsz = text_emb.size(0)
            if (
                text_emb.abs().sum() == 0
                and video_emb.abs().sum() == 0
                and audio_emb.abs().sum() == 0
            ):
                return torch.zeros(bsz, self.mlp[-1].out_features, device=text_emb.device)

            if text_emb.abs().sum() == 0:
                text_emb = self.missing_text.unsqueeze(0).expand(bsz, -1)
            if video_emb.abs().sum() == 0:
                video_emb = self.missing_video.unsqueeze(0).expand(bsz, -1)
            if audio_emb.abs().sum() == 0:
                audio_emb = self.missing_audio.unsqueeze(0).expand(bsz, -1)

            t_h = self.proj_t(text_emb)
            v_h = self.proj_v(video_emb)
            a_h = self.proj_a(audio_emb)
            fused = (t_h + v_h + a_h) / 3.0
            return self.mlp(fused)

    class ConcatProjections(nn.Module):
        """C1 v2: Remove co-attention entirely, concat projected features -> MLP.

        Preserves projections and missing embeddings, but removes ALL
        cross-modal co-attention. Simply concat 3 projected vectors and
        pass through MLP. Tests whether the co-attention mechanism is
        the essential component of Module 1.
        """

        def __init__(
            self,
            imagebind_dim: int = 1024,
            proj_dim: int = 256,
            hidden_dim: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.proj_t = nn.Linear(imagebind_dim, proj_dim)
            self.proj_v = nn.Linear(imagebind_dim, proj_dim)
            self.proj_a = nn.Linear(imagebind_dim, proj_dim)
            self.mlp = nn.Sequential(
                nn.Linear(proj_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.missing_text = nn.Parameter(torch.zeros(imagebind_dim))
            self.missing_video = nn.Parameter(torch.zeros(imagebind_dim))
            self.missing_audio = nn.Parameter(torch.zeros(imagebind_dim))
            self._init_weights()

        def _init_weights(self) -> None:
            for p in self.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)

        def forward(
            self, text_emb: torch.Tensor, video_emb: torch.Tensor, audio_emb: torch.Tensor
        ) -> torch.Tensor:
            bsz = text_emb.size(0)
            if (
                text_emb.abs().sum() == 0
                and video_emb.abs().sum() == 0
                and audio_emb.abs().sum() == 0
            ):
                return torch.zeros(bsz, self.mlp[-1].out_features, device=text_emb.device)

            if text_emb.abs().sum() == 0:
                text_emb = self.missing_text.unsqueeze(0).expand(bsz, -1)
            if video_emb.abs().sum() == 0:
                video_emb = self.missing_video.unsqueeze(0).expand(bsz, -1)
            if audio_emb.abs().sum() == 0:
                audio_emb = self.missing_audio.unsqueeze(0).expand(bsz, -1)

            t_h = self.proj_t(text_emb)
            v_h = self.proj_v(video_emb)
            a_h = self.proj_a(audio_emb)
            return self.mlp(torch.cat([t_h, v_h, a_h], dim=-1))

    class FactBranchMLP(nn.Module):
        """Process 7x768 BERT CLS vectors -> FEXT_FACT(256).

        ARCHITECTURE.md x4.2.6
        """

        def __init__(
            self,
            num_branches: int = 7,
            bert_dim: int = 768,
            hidden_dim: int = 1024,
            output_dim: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.num_branches = num_branches
            self.net = nn.Sequential(
                nn.Linear(num_branches * bert_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, branch_outputs: torch.Tensor) -> torch.Tensor:
            B = branch_outputs.size(0)
            return self.net(branch_outputs.view(B, -1))

    class SimpleFactBranchMLP(nn.Module):
        """C2: Average across entity branches instead of flatten+MLP.

        Input (B,7,768) -> avg over dim=1 -> (B,768) -> Linear(768,256) -> ReLU.
        """

        def __init__(self, bert_dim: int = 768, output_dim: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(bert_dim, output_dim),
                nn.ReLU(),
            )

        def forward(self, branch_outputs: torch.Tensor) -> torch.Tensor:
            return self.net(branch_outputs.mean(dim=1))

    class EntityConditionedFactBranchMLP(nn.Module):
        """Module 2 with entity-conditioned branch fusion.

        Replaces the flatten -> Linear aggregation of FactBranchMLP with
        entity-conditioned attention weighting. Entity statistics (5 stats x
        7 types = 35-dim) are used to compute per-branch contribution weights,
        enabling the model to dynamically prioritize entity types based on
        their retrieval quality statistics.

        The forward method supports an `ablate_gating` flag: when True, all
        branches receive uniform weight, disabling the ES conditioning while
        keeping the rest of the architecture identical. This provides a clean
        ablation test for the contribution of entity statistics.

        Reference: ARCHITECTURE.md x4 (extended)
        """

        def __init__(
            self,
            num_branches: int = 7,
            bert_dim: int = 768,
            hidden_dim: int = 1024,
            output_dim: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.num_branches = num_branches
            self.branch_encoder = nn.Sequential(
                nn.Linear(bert_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            # ES -> per-branch weight
            self.es_gate = nn.Sequential(
                nn.Linear(35, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_branches),
                nn.Softmax(dim=-1),
            )
            self.output_proj = nn.Linear(output_dim, output_dim)
            self._init_weights()

        def _init_weights(self) -> None:
            for p in self.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)

        def forward(
            self,
            branch_outputs: torch.Tensor,
            entity_stats: torch.Tensor,
            ablate_gating: bool = False,
        ) -> torch.Tensor:
            B = branch_outputs.size(0)
            encoded = self.branch_encoder(branch_outputs.view(B * self.num_branches, -1))
            encoded = encoded.view(B, self.num_branches, -1)
            weights = self.es_gate(entity_stats)
            if ablate_gating:
                weights = torch.ones_like(weights) / self.num_branches
            fused = (weights.unsqueeze(-1) * encoded).sum(dim=1)
            return self.output_proj(fused)

    class StyleMLP(nn.Module):
        """BERT CLS(768) -> MLP_3 -> FSTYLE(256).

        ARCHITECTURE.md x5.2.4
        """

        def __init__(
            self,
            bert_dim: int = 768,
            hidden_dim: int = 256,
            output_dim: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(bert_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, bert_cls: torch.Tensor) -> torch.Tensor:
            return self.net(bert_cls)

    class SimpleStyleMLP(nn.Module):
        """C3: Single linear projection (768->256). No hidden layer, no activation."""

        def __init__(self, bert_dim: int = 768, output_dim: int = 256) -> None:
            super().__init__()
            self.net = nn.Linear(bert_dim, output_dim)

        def forward(self, bert_cls: torch.Tensor) -> torch.Tensor:
            return self.net(bert_cls)

    class TextAuxCompressor(nn.Module):
        """Compress hashing(text_aux_dim - 12) + 12-dim stats -> text_aux.

        ARCHITECTURE.md x6.3

        Input dimension = hashing_dim + 12 (text_stats), not hardcoded 524,
        so that config changes to hashing_dim propagate correctly.
        """

        def __init__(self, input_dim: int = 524, output_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class EntityStatsCompressor(nn.Module):
        """Compress 35-dim entity statistics -> entity_feat_dim.

        5 statistics per entity type x 7 types = 35 dims:
          - num_original, num_retrieved, overlap_rate, conflict_proxy, has_entity
        """

        def __init__(self, input_dim: int = 35, output_dim: int = 16) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class FactFusionMLP(nn.Module):
        """Stage 1: FINT_FACT(256) + FEXT_FACT(256) -> FFACT(256).

        ARCHITECTURE.md x6.2

        When extra_layer=True, adds Dropout+Linear(hidden->hidden) for more capacity.
        """

        def __init__(
            self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1,
            extra_layer: bool = False,
        ) -> None:
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            if extra_layer:
                layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)])
            self.net = nn.Sequential(*layers)

        def forward(
            self, fint_fact: torch.Tensor, fext_fact: torch.Tensor
        ) -> torch.Tensor:
            return self.net(torch.cat([fint_fact, fext_fact], dim=-1))

    class StyleAwareClassifier(nn.Module):
        """Stage 2: FFACT(256) + FSTYLE(256) + text_aux(64) + entity_feat(16) -> fake probability.

        ARCHITECTURE.md x6.2
        """

        def __init__(
            self,
            input_dim: int = 576,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            entity_feat_dim: int = 0,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim + entity_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            ffact: torch.Tensor,
            fstyle: torch.Tensor,
            text_aux: torch.Tensor,
            entity_feat: Optional[torch.Tensor] = None,
            extra_feat: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            parts = [ffact, fstyle, text_aux]
            if entity_feat is not None:
                parts.append(entity_feat)
            if extra_feat is not None:
                parts.append(extra_feat)
            return self.net(torch.cat(parts, dim=-1)).squeeze(-1)

    class PerLangStyleAwareClassifier(nn.Module):
        """Style-aware classifier with per-language classification heads.

        Shares backbone between languages but has separate final linear layers
        for zh (SV) and en (TT), enabling language-specific decision boundaries.
        The backbone learns shared features; each head learns language-specific
        classification weights.

        Input: same as StyleAwareClassifier, plus lang_ids for head routing.
        """

        def __init__(
            self,
            input_dim: int = 576,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            entity_feat_dim: int = 0,
        ) -> None:
            super().__init__()
            backbone_dim = input_dim + entity_feat_dim
            self.backbone = nn.Sequential(
                nn.Linear(backbone_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head_zh = nn.Linear(hidden_dim, 1)
            self.head_en = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            ffact: torch.Tensor,
            fstyle: torch.Tensor,
            text_aux: torch.Tensor,
            entity_feat: Optional[torch.Tensor] = None,
            extra_feat: Optional[torch.Tensor] = None,
            lang_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            parts = [ffact, fstyle, text_aux]
            if entity_feat is not None:
                parts.append(entity_feat)
            if extra_feat is not None:
                parts.append(extra_feat)
            h = self.backbone(torch.cat(parts, dim=-1))
            logits_zh = self.head_zh(h)
            logits_en = self.head_en(h)
            if lang_ids is None:
                return (logits_zh + logits_en) / 2.0
            out = torch.where(lang_ids.unsqueeze(-1) == 0, logits_zh, logits_en)
            return out.squeeze(-1)

    class SimpleClassifier(nn.Module):
        """C4: One-stage concat-all -> MLP -> logit.

        Replaces FactFusionMLP + StyleAwareClassifier.
        Input: concat(fint, fext, fstyle, text_aux, [entity_feat]) = 832(+entity) dims.
        """

        def __init__(
            self,
            input_dim: int = 832,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            entity_feat_dim: int = 0,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim + entity_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            fint_fact: torch.Tensor,
            fext_fact: torch.Tensor,
            fstyle: torch.Tensor,
            text_aux: torch.Tensor,
            entity_feat: Optional[torch.Tensor] = None,
            extra_feat: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            parts = [fint_fact, fext_fact, fstyle, text_aux]
            if entity_feat is not None:
                parts.append(entity_feat)
            if extra_feat is not None:
                parts.append(extra_feat)
            return self.net(torch.cat(parts, dim=-1)).squeeze(-1)

    class ResidualFusionBlock(nn.Module):
        """Pre-activation residual MLP block.

        LayerNorm -> GELU -> Linear -> Dropout, with residual connection.
        """

        def __init__(self, dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.gelu = nn.GELU()
            self.linear = nn.Linear(dim, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.drop(self.linear(self.gelu(self.ln(x))))

    class DeepResidualFusion(nn.Module):
        """Deep residual fusion network.

        Projects concatenated features to hidden_dim, passes through
        N residual blocks, then LayerNorm.
        """

        def __init__(
            self,
            input_dim: int = 832,
            hidden_dim: int = 512,
            num_blocks: int = 4,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                [ResidualFusionBlock(hidden_dim, dropout) for _ in range(num_blocks)]
            )
            self.ln = nn.LayerNorm(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h)
            return self.ln(h)

    class MultiHeadClassifier(nn.Module):
        """Multi-head classification layer.

        N independent classification heads, each: Linear(128) -> GELU -> Dropout -> Linear(1).
        At training: each head supervised independently, loss averaged.
        At inference: all head outputs averaged.
        """

        def __init__(
            self, input_dim: int = 512, num_heads: int = 4, dropout: float = 0.1
        ) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1),
                )
                for _ in range(num_heads)
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outputs = [head(x) for head in self.heads]
            return torch.stack(outputs, dim=-1).mean(dim=-1).squeeze(-1)

    class LoRALinear(nn.Module):
        """Low-rank adaptation wrapper for nn.Linear.

        For a frozen pretrained weight W, forward becomes:
            y = Wx + (x @ A @ B) * (alpha / r)
        where A (in_features x r) and B (r x out_features) are the only
        trainable parameters. Typically r=4..32, alpha=16.

        Reference: Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022.
        """

        def __init__(self, original: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
            super().__init__()
            self.original = original
            self.r = r
            self.alpha = alpha
            self.scaling = alpha / r
            in_features = original.in_features
            out_features = original.out_features
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            self.original.requires_grad_(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

    class GradientReversal(torch.autograd.Function):
        """Gradient reversal layer for domain adversarial training.

        Forward: identity (pass input through unchanged).
        Backward: scale gradients by -alpha (reverse direction).
        """

        @staticmethod
        def forward(ctx, x, alpha: float = 1.0) -> torch.Tensor:
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
            return grad_output.neg() * ctx.alpha, None

    def gradient_rev(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Apply gradient reversal with scaling factor alpha."""
        return GradientReversal.apply(x, alpha)

    class FactStyleFusionClassifier(nn.Module):
        """End-to-end fake news detection model.

        ARCHITECTURE.md x2-x6

        During Stage 1 (cache building), this model is used for GPU-bound
        operations (ImageBind, BERT). During Stage 2 (training), only the
        MLP and fusion parameters are trained from cached intermediate
        representations.
        """

        def __init__(
            self,
            imagebind_dim: int = 1024,
            proj_dim: int = 256,
            bert_dim: int = 768,
            num_fact_branches: int = 7,
            mlp2_hidden: int = 1024,
            module_output_dim: int = 256,
            text_aux_dim: int = 64,
            text_aux_input_dim: int = 524,
            dropout: float = 0.1,
            entity_stats_dim: int = 0,
            ablate_module1: bool = False,
            ablate_module2: bool = False,
            ablate_module3: bool = False,
            ablate_text_aux: bool = False,
            simple_module1: bool = False,
            simple_module2: bool = False,
            simple_module3: bool = False,
            simple_fusion: bool = False,
            weak_module1: bool = False,
            no_co_attn: bool = False,
            weak_fusion: bool = False,
            entity_conditioned: bool = False,
            ablate_entity_gating: bool = False,
            deep_fusion: bool = False,
            num_heads: int = 4,
            train_bert: bool = False,
            bert_tune_layers: int = 0,
            lora_rank: int = 0,
            bert_pool: str = "cls",
            language_aware: bool = False,
            lang_emb_dim: int = 0,
            per_lang_classifier: bool = False,
            dann_alpha: float = 0.0,
            fact_fusion_extra: bool = False,
        ) -> None:
            super().__init__()
            self.ablate_module1 = ablate_module1
            self.ablate_module2 = ablate_module2
            self.ablate_module3 = ablate_module3
            self.ablate_text_aux = ablate_text_aux
            self.simple_module1 = simple_module1
            self.simple_module2 = simple_module2
            self.simple_module3 = simple_module3
            self.simple_fusion = simple_fusion
            self.weak_module1 = weak_module1
            self.no_co_attn = no_co_attn
            self.weak_fusion = weak_fusion
            self.entity_conditioned = entity_conditioned
            self.ablate_entity_gating = ablate_entity_gating
            self.deep_fusion = deep_fusion
            self.num_heads = num_heads
            self.language_aware = language_aware
            self.per_lang_classifier = per_lang_classifier
            self.dann_alpha = dann_alpha
            self.train_bert = train_bert
            self.fact_fusion_extra = fact_fusion_extra
            self.bert_tune_layers = bert_tune_layers
            self.lora_rank = lora_rank
            self.bert_pool = bert_pool
            self.module_output_dim = module_output_dim

            # --- Module 1: multimodal consistency ---
            if simple_module1:
                self.module1 = SimpleTriModalConsistency(
                    imagebind_dim=imagebind_dim, proj_dim=proj_dim,
                    hidden_dim=module_output_dim, dropout=dropout,
                )
            elif no_co_attn:
                self.module1 = ConcatProjections(
                    imagebind_dim=imagebind_dim, proj_dim=proj_dim,
                    hidden_dim=module_output_dim, dropout=dropout,
                )
            else:
                m1_num_heads = 1 if weak_module1 else 4
                self.module1 = GatedTriModalConsistency(
                    imagebind_dim=imagebind_dim, proj_dim=proj_dim,
                    hidden_dim=module_output_dim,
                    num_heads=m1_num_heads, dropout=dropout,
                )

            # --- Module 2: external fact deviation ---
            if entity_conditioned:
                self.module2_mlp = EntityConditionedFactBranchMLP(
                    num_branches=num_fact_branches, bert_dim=bert_dim,
                    hidden_dim=mlp2_hidden, output_dim=module_output_dim,
                    dropout=dropout,
                )
            elif simple_module2:
                self.module2_mlp = SimpleFactBranchMLP(
                    bert_dim=bert_dim, output_dim=module_output_dim,
                )
            else:
                self.module2_mlp = FactBranchMLP(
                    num_branches=num_fact_branches, bert_dim=bert_dim,
                    hidden_dim=mlp2_hidden, output_dim=module_output_dim,
                    dropout=dropout,
                )

            # Absent entity embeddings (Module 2)
            self.absent_embeddings = nn.Parameter(
                torch.zeros(num_fact_branches, bert_dim)
            )
            nn.init.normal_(self.absent_embeddings, std=0.02)

            # --- Module 3: style/skeleton deviation ---
            if simple_module3:
                self.module3_mlp = SimpleStyleMLP(
                    bert_dim=bert_dim, output_dim=module_output_dim,
                )
            else:
                self.module3_mlp = StyleMLP(
                    bert_dim=bert_dim, hidden_dim=module_output_dim,
                    output_dim=module_output_dim, dropout=dropout,
                )

            # --- Trainable BERT (LoRA / partial tuning) ---
            self.bert_model = None
            if train_bert and not simple_module3:
                from transformers import BertModel
                self.bert_model = BertModel.from_pretrained(
                    "models/bert-base-multilingual-cased"
                )
                for param in self.bert_model.parameters():
                    param.requires_grad = False

                if lora_rank > 0:
                    for layer in self.bert_model.encoder.layer:
                        layer.attention.self.query = LoRALinear(
                            layer.attention.self.query, r=lora_rank, alpha=16.0,
                        )
                        layer.attention.self.key = LoRALinear(
                            layer.attention.self.key, r=lora_rank, alpha=16.0,
                        )
                        layer.attention.self.value = LoRALinear(
                            layer.attention.self.value, r=lora_rank, alpha=16.0,
                        )
                        layer.attention.output.dense = LoRALinear(
                            layer.attention.output.dense, r=lora_rank, alpha=16.0,
                        )
                    n_lora = sum(
                        1 for n, _ in self.bert_model.named_parameters()
                        if "lora_" in n
                    )
                    n_total = self.bert_model.num_parameters() / 1e6
                    print(
                        f"  BERT: LoRA (rank={lora_rank}, {n_lora} "
                        f"trainable adapters, all {n_total:.0f}M frozen)"
                    )
                elif bert_tune_layers > 0:
                    for layer in self.bert_model.encoder.layer[-bert_tune_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    if bert_tune_layers >= 8:
                        for param in self.bert_model.embeddings.parameters():
                            param.requires_grad = True
                    n_total = self.bert_model.num_parameters() / 1e6
                    print(
                        f"  BERT: trainable ({n_total:.0f}M params, "
                        f"tuning top {bert_tune_layers} layers)"
                    )
                elif bert_tune_layers == 0:
                    for param in self.bert_model.parameters():
                        param.requires_grad = True
                    n_total = self.bert_model.num_parameters() / 1e6
                    print(
                        f"  BERT: trainable ({n_total:.0f}M params, "
                        f"tuning top {bert_tune_layers} layers)"
                    )

                if bert_pool == "mean":
                    print("  BERT: mean pooling over all tokens")

            # --- Text auxiliary compressor ---
            self.text_aux_compressor = TextAuxCompressor(
                input_dim=text_aux_input_dim, output_dim=text_aux_dim,
            )

            # --- Entity stats ---
            if entity_conditioned:
                effective_es_dim = 0
            else:
                effective_es_dim = entity_stats_dim
            self.entity_stats_dim = effective_es_dim

            if effective_es_dim > 0:
                self.entity_stats_compressor = EntityStatsCompressor(
                    input_dim=35, output_dim=effective_es_dim,
                )
            else:
                self.entity_stats_compressor = None

            # --- Language embedding ---
            if language_aware:
                self.lang_emb_dim = lang_emb_dim
            else:
                self.lang_emb_dim = 0
            if language_aware:
                self.lang_embedding = nn.Embedding(2, self.lang_emb_dim)
            else:
                self.lang_embedding = None

            # --- Fusion & Classifier ---
            if deep_fusion:
                df_input_dim = (
                    module_output_dim * 3 + text_aux_dim
                    + effective_es_dim + self.lang_emb_dim
                )
                self.deep_residual_fusion = DeepResidualFusion(
                    input_dim=df_input_dim, hidden_dim=512,
                    num_blocks=4, dropout=dropout,
                )
                self.multi_head_classifier = MultiHeadClassifier(
                    input_dim=512, num_heads=num_heads, dropout=dropout,
                )
            elif simple_fusion:
                fusion_input_dim = (
                    module_output_dim * 3 + text_aux_dim + self.lang_emb_dim
                )
                self.simple_classifier = SimpleClassifier(
                    input_dim=fusion_input_dim, hidden_dim=module_output_dim,
                    dropout=dropout, entity_feat_dim=effective_es_dim,
                )
            else:
                fusion_hidden_dim = 16 if weak_fusion else module_output_dim
                self.fact_fusion = FactFusionMLP(
                    input_dim=module_output_dim * 2,
                    hidden_dim=module_output_dim, dropout=dropout,
                    extra_layer=self.fact_fusion_extra,
                )
                classifier_input_dim = (
                    module_output_dim + module_output_dim
                    + text_aux_dim + self.lang_emb_dim
                )
                if per_lang_classifier:
                    self.classifier = PerLangStyleAwareClassifier(
                        input_dim=classifier_input_dim,
                        hidden_dim=fusion_hidden_dim,
                        dropout=dropout,
                        entity_feat_dim=effective_es_dim,
                    )
                else:
                    self.classifier = StyleAwareClassifier(
                        input_dim=classifier_input_dim,
                        hidden_dim=fusion_hidden_dim,
                        dropout=dropout,
                        entity_feat_dim=effective_es_dim,
                    )

            # --- DANN: language discriminator ---
            if dann_alpha > 0:
                dann_input_dim = module_output_dim * 3 + text_aux_dim
                if not entity_conditioned and effective_es_dim > 0:
                    dann_input_dim += effective_es_dim
                self.lang_discriminator = nn.Sequential(
                    nn.Linear(dann_input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 2),
                )
                self._dann_feat: Optional[torch.Tensor] = None

            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward_module1(
            self, text_emb: torch.Tensor, video_emb: torch.Tensor, audio_emb: torch.Tensor
        ) -> torch.Tensor:
            if self.ablate_module1:
                return torch.zeros(
                    text_emb.size(0), self.module_output_dim, device=text_emb.device
                )
            return self.module1(text_emb, video_emb, audio_emb)

        def forward_module2(
            self,
            branch_bert_cls: torch.Tensor,
            branch_mask: torch.Tensor,
            entity_stats: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if self.ablate_module2:
                return torch.zeros(
                    branch_bert_cls.size(0), self.module_output_dim,
                    device=branch_bert_cls.device,
                )
            bsz = branch_bert_cls.size(0)
            absent = self.absent_embeddings.unsqueeze(0).expand(bsz, -1, -1)
            mask = branch_mask.unsqueeze(-1).float()
            branch_outputs = branch_bert_cls * mask + absent * (1 - mask)
            if self.entity_conditioned:
                return self.module2_mlp(
                    branch_outputs, entity_stats=entity_stats,
                    ablate_gating=self.ablate_entity_gating,
                )
            return self.module2_mlp(branch_outputs)

        def forward_module3(
            self,
            style_bert_cls: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if self.ablate_module3:
                device = style_bert_cls.device if style_bert_cls is not None else input_ids.device
                batch = (
                    style_bert_cls.size(0) if style_bert_cls is not None
                    else input_ids.size(0)
                )
                return torch.zeros(batch, self.module_output_dim, device=device)

            if self.train_bert and self.bert_model is not None:
                outputs = self.bert_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                if self.bert_pool == "mean":
                    token_emb = outputs.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).float()
                    sum_emb = (token_emb * mask).sum(dim=1)
                    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = sum_emb / sum_mask
                else:
                    pooled = outputs.last_hidden_state[:, 0]
                return self.module3_mlp(pooled)

            return self.module3_mlp(style_bert_cls)

        def forward_entity_stats(self, stats: torch.Tensor) -> Optional[torch.Tensor]:
            if self.entity_stats_compressor is None:
                return None
            return self.entity_stats_compressor(stats)

        def forward_fusion(
            self,
            fint_fact: torch.Tensor,
            fext_fact: torch.Tensor,
            fstyle: torch.Tensor,
            text_aux_features: torch.Tensor,
            entity_stats: Optional[torch.Tensor] = None,
            lang_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if self.ablate_text_aux:
                text_aux = torch.zeros(
                    fint_fact.size(0), 64, device=fint_fact.device
                )
            else:
                text_aux = self.text_aux_compressor(text_aux_features)

            entity_feat = (
                self.forward_entity_stats(entity_stats)
                if entity_stats is not None else None
            )

            lang_feat = None
            if self.lang_embedding is not None:
                if lang_ids is None:
                    lang_ids = torch.zeros(
                        fint_fact.size(0), dtype=torch.long, device=fint_fact.device
                    )
                lang_feat = self.lang_embedding(lang_ids.long())

            if self.deep_fusion:
                parts = [fint_fact, fext_fact, fstyle, text_aux]
                if lang_feat is not None:
                    parts.append(lang_feat)
                if entity_feat is not None:
                    parts.append(entity_feat)
                combined = torch.cat(parts, dim=-1)
                feat = self.deep_residual_fusion(combined)
                logits = self.multi_head_classifier(feat)
            elif self.simple_fusion:
                logits = self.simple_classifier(
                    fint_fact, fext_fact, fstyle, text_aux,
                    entity_feat=entity_feat, extra_feat=lang_feat,
                )
            else:
                ffact = self.fact_fusion(fint_fact, fext_fact)  # (batch, 256)
                classifier_kwargs = dict(
                    entity_feat=entity_feat, extra_feat=lang_feat,
                )
                if self.per_lang_classifier:
                    classifier_kwargs["lang_ids"] = lang_ids
                logits = self.classifier(
                    ffact, fstyle, text_aux, **classifier_kwargs
                )

            # Cache features for DANN language discriminator (training only)
            if self.dann_alpha > 0 and self.training:
                if self.deep_fusion:
                    self._dann_feat = feat
                else:
                    dann_parts = [fint_fact, fext_fact, fstyle, text_aux]
                    if entity_feat is not None:
                        dann_parts.append(entity_feat)
                    self._dann_feat = torch.cat(dann_parts, dim=-1)

            return logits

        def get_dann_logits(
            self, lang_ids: torch.Tensor
        ) -> Optional[torch.Tensor]:
            """Return language discriminator logits for DANN training.

            Must be called after forward_fusion during training when dann_alpha > 0.
            The gradient reversal layer reverses gradients flowing into the fusion
            features, removing language information from the shared representation.
            """
            if not self.training or self.dann_alpha <= 0 or self._dann_feat is None:
                return None
            feat = gradient_rev(self._dann_feat, alpha=self.dann_alpha)
            self._dann_feat = None
            return self.lang_discriminator(feat)

        def forward(
            self,
            text_emb: torch.Tensor,
            video_emb: torch.Tensor,
            audio_emb: torch.Tensor,
            branch_bert_cls: torch.Tensor,
            branch_mask: torch.Tensor,
            style_bert_cls: torch.Tensor,
            text_aux_features: torch.Tensor,
            entity_stats: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            fint = self.forward_module1(text_emb, video_emb, audio_emb)
            fext = self.forward_module2(
                branch_bert_cls, branch_mask, entity_stats=entity_stats,
            )
            fstyle = self.forward_module3(style_bert_cls)
            return self.forward_fusion(
                fint, fext, fstyle, text_aux_features,
                entity_stats=entity_stats,
            )

else:

    # Stub classes for environments without PyTorch
    class CoAttentionBlock:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for CoAttentionBlock")

    class GatedTriModalConsistency:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for GatedTriModalConsistency")

    class SimpleTriModalConsistency:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for SimpleTriModalConsistency")

    class ConcatProjections:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for ConcatProjections")

    class FactBranchMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for FactBranchMLP")

    class SimpleFactBranchMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for SimpleFactBranchMLP")

    class EntityConditionedFactBranchMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for EntityConditionedFactBranchMLP")

    class StyleMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for StyleMLP")

    class SimpleStyleMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for SimpleStyleMLP")

    class TextAuxCompressor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TextAuxCompressor")

    class EntityStatsCompressor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for EntityStatsCompressor")

    class FactFusionMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for FactFusionMLP")

    class StyleAwareClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for StyleAwareClassifier")

    class PerLangStyleAwareClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for PerLangStyleAwareClassifier")

    class SimpleClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for SimpleClassifier")

    class ResidualFusionBlock:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for ResidualFusionBlock")

    class DeepResidualFusion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for DeepResidualFusion")

    class MultiHeadClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for MultiHeadClassifier")

    class LoRALinear:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for LoRALinear")

    class GradientReversal:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for GradientReversal")

    def gradient_rev(x, alpha=1.0):  # type: ignore
        raise ImportError("PyTorch is required for gradient_rev")

    class FactStyleFusionClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for FactStyleFusionClassifier")
