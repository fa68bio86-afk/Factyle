"""Text feature extraction: hashing bag-of-words + shallow linguistic stats.

Reference: ARCHITECTURE.md §6.3 (text_aux auxiliary signal)
"""

import math
import re
from typing import List

import numpy as np


class HashingVectorizer:
    """Hashing bag-of-words feature (ARCHITECTURE.md §6.3).

    Produces hashing_dim (default 512) sparse binary features.
    """

    def __init__(self, hashing_dim: int = 512, ngram_range: tuple = (1, 2)):
        self.hashing_dim = hashing_dim
        self.ngram_range = ngram_range

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens

    def _hash_feature(self, token: str) -> int:
        """Deterministic hash to [0, hashing_dim)."""
        return abs(hash(token)) % self.hashing_dim

    def transform(self, text: str) -> np.ndarray:
        """Transform text to hashing feature vector."""
        vec = np.zeros(self.hashing_dim, dtype=np.float32)
        tokens = self._tokenize(text)
        for token in tokens:
            idx = self._hash_feature(token)
            vec[idx] += 1.0
        # Add bigrams
        if self.ngram_range[1] >= 2:
            for i in range(len(tokens) - 1):
                bigram = tokens[i] + "_" + tokens[i + 1]
                idx = self._hash_feature(bigram)
                vec[idx] += 1.0
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class TextStatsExtractor:
    """12-dim shallow linguistic features (ARCHITECTURE.md §6.3 text_stats).

    | # | Feature         | Calculation                         |
    |---|-----------------|-------------------------------------|
    | 1 | chars_norm      | chars / 2000 (capped at 1.0)        |
    | 2 | words_norm      | words / 400 (capped at 1.0)         |
    | 3 | digit_ratio     | digits / chars                      |
    | 4 | cjk_ratio       | CJK chars / chars                   |
    | 5 | exclaim_norm    | exclaim / 10 (capped at 1.0)        |
    | 6 | question_norm   | question / 10 (capped at 1.0)       |
    | 7 | comma_norm      | comma / 50 (capped at 1.0)          |
    | 8 | quotes_norm     | quotes / 20 (capped at 1.0)         |
    | 9 | lexical_diversity | unique tokens / words              |
    | 10 | upper_ratio     | uppercase chars / chars             |
    | 11 | sensational_norm | sensational markers / 5 (capped 1.0)|
    | 12 | official_norm   | official markers / 5 (capped 1.0)   |
    """

    # Chinese sensational/clickbait markers commonly found in fake news
    SENSATIONAL_MARKERS = {
        "震惊", "惊曝", "太可怕", "难以置信", "出大事", "紧急", "疯狂", "真相是",
        "shocking", "unbelievable", "terrifying", "mind-blowing", "you won't believe",
        "insane", "crazy", "explosive", "outrageous", "heartbreaking",
    }
    # Chinese official/authoritative language markers
    OFFICIAL_MARKERS = {
        "据通报", "官方", "警方", "通报", "经核实", "据调查", "相关部门",
        "according to", "official", "police", "authorities", "confirmed",
        "government", "reported by",
    }

    def extract(self, text: str) -> np.ndarray:
        """Return 12-dim numpy array."""
        chars = len(text)
        tokens = re.findall(r"\w+", text)
        words = len(tokens) if tokens else 1  # avoid division by zero

        # Basic counts
        digits = sum(c.isdigit() for c in text)
        cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
        exclaim = text.count("!") + text.count("！")
        question = text.count("?") + text.count("？")
        commas = text.count(",") + text.count("，")
        quotes = text.count('"') + text.count("'") + text.count("「") + text.count("『")
        uppercase = sum(c.isupper() for c in text)

        # Lexical diversity
        unique_tokens = len(set(t.lower() for t in tokens))
        lexical_diversity = unique_tokens / words if words > 0 else 0

        # Sensational / official markers
        text_lower = text.lower()
        sensational_count = sum(1 for m in self.SENSATIONAL_MARKERS if m in text_lower)
        official_count = sum(1 for m in self.OFFICIAL_MARKERS if m in text_lower)

        stats = np.array(
            [
                min(chars / 2000.0, 1.0),
                min(words / 400.0, 1.0),
                digits / max(chars, 1),
                cjk / max(chars, 1),
                min(exclaim / 10.0, 1.0),
                min(question / 10.0, 1.0),
                min(commas / 50.0, 1.0),
                min(quotes / 20.0, 1.0),
                lexical_diversity,
                uppercase / max(chars, 1),
                min(sensational_count / 5.0, 1.0),
                min(official_count / 5.0, 1.0),
            ],
            dtype=np.float32,
        )
        return stats


def build_text_aux(text: str, hashing_dim: int = 512) -> np.ndarray:
    """Build combined hashing + text_stats feature (hashing_dim + 12).

    Input to the text_aux compression network: Linear(hashing_dim + 12, 64) → ReLU.
    """
    hashing = HashingVectorizer(hashing_dim=hashing_dim).transform(text)
    stats = TextStatsExtractor().extract(text)
    return np.concatenate([hashing, stats])
