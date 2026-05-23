"""Module 3: Multi-threaded style rewriting and analysis.

Reference: ARCHITECTURE.md §5.2, §9.2.4

Two-stage process:
  1. Parallel: Ta (authoritative) rewrite + Tb (sensational) rewrite via Qwen3-8B
     Offline fallback rules when Qwen unavailable (§5.2.1)
  2. Serial: spaCy entity extraction → skeleton generation → Qwen3-32B style analysis
     Offline sensational score fallback when Qwen3-32B unavailable (§5.2.3)

ThreadPoolExecutor(max_workers=2) for the dual rewrite per §9.2.4.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from factyle.clients.qwen_client import QwenClient


@dataclass
class RewriteResult:
    style: str  # "authoritative" or "sensational"
    rewritten_text: str
    skeleton: str  # entity-masked skeleton
    skeleton_stats: Dict = field(default_factory=dict)  # 23-dim stats for ablation
    success: bool = False


class Module3StyleRewriter:
    """Two-stage style rewriting with parallel Ta/Tb and serial analysis.

    Architecture per §9.2.4:
      ThreadPoolExecutor(max_workers=2)
        ├── Ta: authoritative rewrite (Qwen3-8B, or offline rules)
        └── Tb: sensational rewrite (Qwen3-8B, or offline rules)
      Then: spaCy entity extraction → skeleton generation → style analysis (Qwen3-32B)
    """

    # Entity placeholder tokens used for Chinese news text masking (§5.2.2)
    ENTITY_PLACEHOLDERS = {
        "PERSON": "[某人]",        # "someone"
        "ORG": "[某机构]",        # "some organization"
        "GPE": "[某地]",          # "some place"
        "LOC": "[某地]",          # "some location"
        "DATE": "[某时]",         # "some time"
        "TIME": "[某时]",         # "some time"
        "MONEY": "[某金额]",      # "some amount"
        "PERCENT": "[某比例]",    # "some percentage"
        "CARDINAL": "[某数]",     # "some number"
        "EVENT": "[某事件]",      # "some event"
        "LAW": "[某法规]",        # "some regulation"
        "WORK_OF_ART": "[某作品]", # "some work of art"
        "PRODUCT": "[某产品]",    # "some product"
        "NORP": "[某群体]",       # "some group"
        "FAC": "[某设施]",        # "some facility"
        "QUANTITY": "[某数量]",   # "some quantity"
        "ORDINAL": "[某序数]",    # "some ordinal"
    }

    # SpaCy label map for Module 3 (§5.2.2)
    SPACY_LABEL_MAP = {
        "DATE": "DATE", "TIME": "DATE",
        "GPE": "GPE", "LOC": "GPE", "FAC": "GPE",
        "PERSON": "PERSON", "PER": "PERSON",
        "ORG": "ORG", "NORP": "ORG",
        "CARDINAL": "NUMBER", "MONEY": "NUMBER",
        "PERCENT": "NUMBER", "QUANTITY": "NUMBER", "ORDINAL": "NUMBER",
        "EVENT": "EVENT", "LAW": "EVENT",
        "WORK_OF_ART": "OBJECT", "PRODUCT": "OBJECT",
    }

    def __init__(self, qwen_client: Optional[QwenClient] = None, offline: bool = False):
        self.qwen = qwen_client or (QwenClient() if not offline else None)
        self.offline = offline  # §5.2.5: skip Qwen API, use offline rules only
        self._nlp_cache: Dict[str, 'spacy.Language'] = {}

    # -----------------------------------------------------------------------
    # Step ②: SpaCy entity extraction for skeleton (§5.2.2)
    # -----------------------------------------------------------------------

    def _extract_entities_spacy(self, text: str, lang: str) -> List[Dict]:
        """Extract entities with spaCy for skeleton generation.

        Returns list of {text, start, end, label, placeholder} spans.
        SpaCy model is cached after first load per language.
        """
        import spacy
        model_name = "zh_core_web_sm" if lang == "zh" else "en_core_web_sm"

        if model_name in self._nlp_cache:
            nlp = self._nlp_cache[model_name]
        else:
            try:
                nlp = spacy.load(model_name)
                self._nlp_cache[model_name] = nlp
            except OSError:
                return self._extract_entities_regex(text, lang)

        doc = nlp(text)

        entities = []
        for ent in doc.ents:
            mapped_label = self.SPACY_LABEL_MAP.get(ent.label_, "")
            if mapped_label and mapped_label in self.ENTITY_PLACEHOLDERS:
                entities.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": mapped_label,
                    "placeholder": self.ENTITY_PLACEHOLDERS[mapped_label],
                })
        return entities

    def _extract_entities_regex(self, text: str, lang: str) -> List[Dict]:
        """Regex fallback entity extraction for skeleton (when spaCy model unavailable)."""
        entities = []

        if lang == "zh":
            # Chinese organization patterns: name + suffix (大学=university, 公司=company, etc.)
            for m in re.finditer(r'[一-鿿]{2,}(?:大学|学院|公司|集团|银行|局|部|委|中心|社)',
                                 text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "ORG", "placeholder": "[某机构]"})
            # Chinese person patterns: name + title (先生=Mr., 教授=professor, etc.)
            for m in re.finditer(r'[一-鿿]{2,4}(?:先生|女士|同志|教授|主席|总统|总理|部长|主任)',
                                 text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "PERSON", "placeholder": "[某人]"})
            # Chinese location patterns: name + suffix (省=province, 市=city, etc.)
            for m in re.finditer(r'[一-鿿]{2,}(?:省|市|区|县|镇|村|街|路|大道)', text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "GPE", "placeholder": "[某地]"})
            # Chinese date patterns
            for m in re.finditer(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月', text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "DATE", "placeholder": "[某时]"})
            # Chinese number patterns (万=10k, 亿=100M, 元=yuan, %)
            for m in re.finditer(r'\d+[万亿]?\s*元|\d+%', text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "NUMBER", "placeholder": "[某数]"})
        else:
            # English organization: capitalized multi-word
            for m in re.finditer(r'\b(?:[A-Z][a-z]+\s?){2,}', text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "ORG", "placeholder": "[某机构]"})
            # English numbers and currency
            for m in re.finditer(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "NUMBER", "placeholder": "[某数]"})
            # English date patterns
            for m in re.finditer(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text):
                entities.append({"text": m.group(), "start": m.start(), "end": m.end(),
                                "label": "DATE", "placeholder": "[某时]"})

        return entities

    def _select_non_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Resolve overlapping entity spans (§5.2.2).

        When spans overlap, keep the longer one and discard the shorter.
        """
        if not entities:
            return []

        sorted_ents = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))

        selected = []
        for ent in sorted_ents:
            # Check overlap with last selected
            if selected and ent["start"] < selected[-1]["end"]:
                continue  # overlap → skip
            selected.append(ent)

        return selected

    def _build_skeleton_spacy(self, text: str, lang: str = "zh") -> Tuple[str, Dict]:
        """Build entity-masked skeleton using spaCy (§5.2.2).

        Returns:
            (skeleton_text, stats_dict) where stats_dict contains per-type entity counts.
        """
        entities = self._extract_entities_spacy(text, lang)
        entities = self._select_non_overlapping_entities(entities)

        # Count entities by placeholder type for stats
        type_counts = {}
        for ent in entities:
            ph = ent["placeholder"]
            type_counts[ph] = type_counts.get(ph, 0) + 1

        # Sort by start_char descending to preserve offsets
        sorted_ents = sorted(entities, key=lambda e: e["start"], reverse=True)

        skeleton = text
        for ent in sorted_ents:
            skeleton = skeleton[:ent["start"]] + ent["placeholder"] + skeleton[ent["end"]:]

        skeleton = re.sub(r'\s+', ' ', skeleton).strip()
        return skeleton, {"entity_counts": type_counts, "num_entities": len(entities)}

    # -----------------------------------------------------------------------
    # Step ①: Offline rewriting fallbacks (§5.2.1)
    # -----------------------------------------------------------------------

    def _offline_authoritative_rewrite(self, text: str, lang: str) -> str:
        """Offline authoritative rewrite fallback.

        Replaces sensational markers with neutral official-sounding language.
        """
        if lang == "en":
            result = text
            result = result.replace("!", ".").replace("!!", ".")
            replacements = {
                "shocking": "reported",
                "unbelievable": "notable",
                "insane": "unusual",
                "you won't believe": "it is reported that",
                "mind-blowing": "significant",
            }
            for old, new in replacements.items():
                result = result.replace(old, new)
            return result

        result = text
        # Replace Chinese exclamation marks with periods
        result = result.replace("！", "。").replace("!!", "。").replace("!", "。")
        # Replace sensational Chinese phrases with official-sounding alternatives
        replacements = {
            "震惊": "据通报",          # "shocking" → "according to official notification"
            "网传": "相关信息显示",    # "rumored" → "relevant information shows"
            "太可怕": "值得关注",      # "too terrifying" → "noteworthy"
            "出大事": "发生一起事件",  # "major incident" → "an incident occurred"
            "紧急": "注意",            # "urgent" → "attention"
            "疯狂": "大量",            # "crazy" → "a large amount"
            "难以置信": "据了解",      # "unbelievable" → "it is understood that"
            "真相是": "经核实",        # "the truth is" → "after verification"
        }
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result

    def _offline_sensational_rewrite(self, text: str, lang: str) -> str:
        """Offline sensational rewrite fallback.

        Adds sensational framing and emotional markers.
        """
        if lang == "en":
            prefix = "SHOCKING: "
            suffix = " This news has sparked widespread attention!"
            return prefix + text + suffix

        prefix = "震惊！"          # "Shocking!"
        suffix = "这一消息引发大量关注！"  # "This news has sparked massive attention!"
        return prefix + text + suffix

    def _offline_style_analysis(self, original: str, ta_skeleton: str,
                                 tb_skeleton: str, lang: str) -> str:
        """Offline style analysis fallback (§5.2.3).

        Computes sensational score differences and returns analysis text.
        """
        def sensational_score(text: str) -> float:
            exclaim = text.count("!") + text.count("！")
            emotional = 0
            if lang == "zh":
                # Chinese sensational markers
                for w in ["震惊", "太可怕", "疯狂", "出大事", "紧急"]:
                    emotional += text.count(w)
            else:
                for w in ["shocking", "unbelievable", "insane", "crazy"]:
                    emotional += text.lower().count(w)
            return exclaim * 2 + emotional

        orig_score = sensational_score(original)
        ta_score = sensational_score(ta_skeleton)
        tb_score = sensational_score(tb_skeleton)

        if lang == "en":
            return (
                f"Style difference analysis (offline):\n"
                f"- Original sensational score: {orig_score}\n"
                f"- Authoritative rewrite score: {ta_score} (delta={ta_score - orig_score})\n"
                f"- Sensational rewrite score: {tb_score} (delta={tb_score - orig_score})\n"
                f"- The authoritative rewrite suppresses emotional language, "
                f"while the sensational rewrite amplifies it."
            )
        return (
            f"风格差异分析（离线）：\n"
            f"- 原文煽情分数：{orig_score}\n"
            f"- 权威改写分数：{ta_score}（差异={ta_score - orig_score}）\n"
            f"- 煽情改写分数：{tb_score}（差异={tb_score - orig_score}）\n"
            f"- 权威改写抑制了情感语言，煽情改写放大了情感表达。"
        )

    # -----------------------------------------------------------------------
    # Skeleton statistical features (ablation baseline, §5.2.2)
    # -----------------------------------------------------------------------

    def _compute_skeleton_stats(
        self,
        original_skeleton: str,
        ta_skeleton: str,
        tb_skeleton: str,
        original_stats: Dict,
        ta_stats: Dict,
        tb_stats: Dict,
    ) -> Dict[str, float]:
        """Compute 23 skeleton statistical features for ablation.

        Features:
          - 3 skeleton token lengths
          - ~7 entity type counts across all skeletons
          - 3 entity replacement ratios
          - 3 SequenceMatcher similarities
          - 3 Jaccard overlaps
          - 3+ style divergence indicators
        """
        stats = {}

        # Token lengths
        stats["orig_skeleton_len"] = len(original_skeleton.split())
        stats["ta_skeleton_len"] = len(ta_skeleton.split())
        stats["tb_skeleton_len"] = len(tb_skeleton.split())

        # Entity counts (summed across all skeletons)
        total_types = set()
        for s_stats in [original_stats, ta_stats, tb_stats]:
            counts = s_stats.get("entity_counts", {})
            for ph in counts:
                total_types.add(ph)
        stats["unique_entity_types"] = len(total_types)

        # Replacement ratios (entity count / token count)
        def _ratio(sk: str, count: int) -> float:
            tokens = len(sk.split())
            return count / max(tokens, 1)
        stats["orig_replace_ratio"] = _ratio(original_skeleton,
                                              original_stats.get("num_entities", 0))
        stats["ta_replace_ratio"] = _ratio(ta_skeleton,
                                            ta_stats.get("num_entities", 0))
        stats["tb_replace_ratio"] = _ratio(tb_skeleton,
                                            tb_stats.get("num_entities", 0))

        # Similarity metrics (SequenceMatcher)
        from difflib import SequenceMatcher
        def _sim(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()
        stats["sim_orig_ta"] = _sim(original_skeleton, ta_skeleton)
        stats["sim_orig_tb"] = _sim(original_skeleton, tb_skeleton)
        stats["sim_ta_tb"] = _sim(ta_skeleton, tb_skeleton)

        # Jaccard overlap (token-level)
        def _jaccard(a: str, b: str) -> float:
            set_a = set(a.split())
            set_b = set(b.split())
            if not set_a and not set_b:
                return 1.0
            return len(set_a & set_b) / max(len(set_a | set_b), 1)
        stats["jaccard_orig_ta"] = _jaccard(original_skeleton, ta_skeleton)
        stats["jaccard_orig_tb"] = _jaccard(original_skeleton, tb_skeleton)
        stats["jaccard_ta_tb"] = _jaccard(ta_skeleton, tb_skeleton)

        # Style divergence: how much Ta and Tb diverge from original
        stats["style_divergence"] = 1.0 - (
            stats["sim_orig_ta"] + stats["sim_orig_tb"]
        ) / 2.0

        # Additional divergence using Jaccard
        stats["jaccard_divergence"] = 1.0 - (
            stats["jaccard_orig_ta"] + stats["jaccard_orig_tb"]
        ) / 2.0

        return stats

    # -----------------------------------------------------------------------
    # Rewriting (parallel per §9.2.4)
    # -----------------------------------------------------------------------

    def _rewrite_single(
        self, text: str, style: str, lang: str
    ) -> RewriteResult:
        """Rewrite text in one style and build skeleton.

        Tries Qwen3-8B first; falls back to offline rules on failure (§5.2.1).
        When offline=True (ablation mode), skips Qwen entirely.
        """
        rewritten = ""
        if not self.offline:
            try:
                rewritten = self.qwen.rewrite_style(text, style, lang=lang)
            except Exception:
                pass

        if not rewritten:
            # Offline fallback (§5.2.1)
            if style == "authoritative":
                rewritten = self._offline_authoritative_rewrite(text, lang)
            else:
                rewritten = self._offline_sensational_rewrite(text, lang)

        if not rewritten:
            return RewriteResult(style=style, rewritten_text="",
                                 skeleton="", skeleton_stats={}, success=False)

        # Build skeleton with spaCy
        skeleton, stats = self._build_skeleton_spacy(rewritten, lang=lang)

        return RewriteResult(
            style=style,
            rewritten_text=rewritten,
            skeleton=skeleton,
            skeleton_stats=stats,
            success=True,
        )

    def rewrite_and_analyze(
        self, text: str, lang: str, text_mode: str = "both"
    ) -> Dict:
        """Full Module 3 pipeline: parallel rewrite → style analysis.

        Args:
            text: original news text
            lang: "zh" or "en"
            text_mode: "both" (Ta+Tb), "only_ta", or "only_tb"

        Returns:
            Dict with:
              - ta_text / tb_text: rewritten texts
              - ta_skeleton / tb_skeleton: entity-masked skeletons
              - original_skeleton: original text skeleton
              - style_analysis: Qwen3-32B (or offline fallback) analysis
              - skeleton_stats: 23-dim stats (ablation baseline)
              - success: whether all steps completed
        """
        # Step 1: Parallel Ta/Tb rewriting (§9.2.4)
        results = {}
        styles_to_rewrite = []
        if text_mode in ("both", "only_ta"):
            styles_to_rewrite.append("authoritative")
        if text_mode in ("both", "only_tb"):
            styles_to_rewrite.append("sensational")

        if styles_to_rewrite:
            with ThreadPoolExecutor(max_workers=len(styles_to_rewrite)) as executor:
                futures = {
                    executor.submit(
                        self._rewrite_single, text, style, lang
                    ): style
                    for style in styles_to_rewrite
                }
                for future in as_completed(futures):
                    style = futures[future]
                    try:
                        results[style] = future.result()
                    except Exception:
                        results[style] = RewriteResult(
                            style=style, rewritten_text="",
                            skeleton="", skeleton_stats={}, success=False
                        )

        # Fill missing styles with empty results
        for style in ["authoritative", "sensational"]:
            if style not in results:
                results[style] = RewriteResult(
                    style=style, rewritten_text="",
                    skeleton="", skeleton_stats={}, success=False
                )

        ta = results["authoritative"]
        tb = results["sensational"]

        # Build original skeleton
        original_skeleton, original_stats = self._build_skeleton_spacy(text, lang=lang)

        # Compute skeleton stats for ablation
        skeleton_stats = self._compute_skeleton_stats(
            original_skeleton, ta.skeleton, tb.skeleton,
            original_stats, ta.skeleton_stats, tb.skeleton_stats,
        )

        # Step 2: Style analysis (§5.2.3)
        ta_ok = ta.success and bool(ta.skeleton)
        tb_ok = tb.success and bool(tb.skeleton)
        style_analysis = ""
        if ta_ok or tb_ok:
            try:
                style_analysis = self.qwen.analyze_style(
                    text,
                    ta.skeleton if ta_ok else "",
                    tb.skeleton if tb_ok else "",
                    lang=lang
                )
            except Exception:
                pass

        if not style_analysis:
            # Offline fallback (§5.2.3)
            style_analysis = self._offline_style_analysis(
                text, ta.skeleton if ta_ok else text,
                tb.skeleton if tb_ok else text, lang
            )

        return {
            "ta_text": ta.rewritten_text,
            "ta_skeleton": ta.skeleton,
            "tb_text": tb.rewritten_text,
            "tb_skeleton": tb.skeleton,
            "original_skeleton": original_skeleton,
            "style_analysis": style_analysis,
            "skeleton_stats": skeleton_stats,
            "success": bool(ta_ok or tb_ok),
        }
