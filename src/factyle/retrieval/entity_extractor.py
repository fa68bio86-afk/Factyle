"""Module 2: Multi-threaded entity extraction with Baidu search + Qwen.

Reference: ARCHITECTURE.md \u00a74.2, \u00a79.2.3

Flow per \u00a74.2.1-\u00a74.2.5:
  1. (Single thread) spaCy NER on original text \u2192 extract entities of 7 types
  2. (Parallel, 7 threads) For each entity type:
     a. If no entities of this type \u2192 TYPE MISSING \u2192 absent_embedding (no search)
     b. If entities exist:
        - Mask entities in text with [MASK_{TYPE}]
        - Filter blocked query terms
        - Baidu search(1 API) \u2192 HTTP fetch(\u22645 URLs) \u2192 Qwen3-8B extraction(1 API)
        - Build branch text with doc: and evidence: fields
  3. BERT CLS per branch (GPU, serial per \u00a79.2.6)

ThreadPoolExecutor(max_workers=7) matches the 7 entity types.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from factyle.clients.baidu_search import BaiduSearchClient
from factyle.clients.qwen_client import QwenClient
from factyle.clients.web_fetcher import WebFetcher


# Entity types matching architecture design (ARCHITECTURE.md \u00a74.2.1)
ENTITY_TYPES = ["time", "location", "person", "organization",
                "number", "event", "object"]

# Blocked query terms to prevent label leakage (\u00a74.2.2)
BLOCKED_QUERY_TERMS = ["label", "fake", "real", "\u771f\u5047", "\u8c23\u8a00",
                       "\u5047\u65b0\u95fb", "fake news", "misinformation", "hoax"]

# -----------------------------------------------------------------------
# Entity normalization maps (\u00a74.2.1)
# -----------------------------------------------------------------------

_LOCATION_ALIASES = {
    # Chinese abbreviations \u2192 full name
    "\u6caa": "\u4e0a\u6d77", "\u7533": "\u4e0a\u6d77", "\u4eac": "\u5317\u4eac", "\u6d25": "\u5929\u6d25", "\u6e1d": "\u91cd\u5e86",
    "\u7ca4": "\u5e7f\u4e1c", "\u95fd": "\u798f\u5efa", "\u6d59": "\u6d59\u6c5f", "\u82cf": "\u6c5f\u82cf", "\u6e58": "\u6e56\u5357",
    "\u9102": "\u6e56\u5317", "\u8c6b": "\u6cb3\u5357", "\u5180": "\u6cb3\u5317", "\u9c81": "\u5c71\u4e1c", "\u664b": "\u5c71\u897f",
    "\u9655": "\u9655\u897f", "\u79e6": "\u9655\u897f", "\u7518": "\u7518\u8083", "\u9647": "\u7518\u8083", "\u8700": "\u56db\u5ddd",
    "\u5ddd": "\u56db\u5ddd", "\u9ed4": "\u8d35\u5dde", "\u6ec7": "\u4e91\u5357", "\u7696": "\u5b89\u5fbd", "\u8d63": "\u6c5f\u897f",
    # English abbreviations \u2192 full name
    "NYC": "New York City", "LA": "Los Angeles", "SF": "San Francisco",
    "DC": "Washington D.C.", "UK": "United Kingdom", "U.K.": "United Kingdom",
    "US": "United States", "USA": "United States", "U.S.": "United States",
    "UAE": "United Arab Emirates", "EU": "European Union",
    "PRC": "China", "HK": "Hong Kong", "SAR": "Hong Kong",
}

_ORG_SUFFIXES = [
    "\u516c\u53f8", "\u96c6\u56e2", "\u80a1\u4efd", "\u6709\u9650", "\u6709\u9650\u8d23\u4efb", "\u6709\u9650\u516c\u53f8", "\u80a1\u4efd\u6709\u9650\u516c\u53f8",
    "\u96c6\u56e2\u80a1\u4efd", "\u603b\u516c\u53f8", "\u5206\u516c\u53f8", "\u5de5\u5382",
    "\u5927\u5b66", "\u5b66\u9662", "\u7814\u7a76\u9662", "\u7814\u7a76\u6240", "\u7814\u7a76\u4e2d\u5fc3", "\u5b9e\u9a8c\u5ba4",
    "\u94f6\u884c", "\u8bc1\u5238", "\u4fdd\u9669", "\u57fa\u91d1", "\u4fe1\u6258",
    "\u5c40", "\u90e8", "\u59d4", "\u529e", "\u5385", "\u5904", "\u6240", "\u4e2d\u5fc3", "\u7f72", "\u603b\u5c40",
    "\u793e", "\u534f\u4f1a", "\u516c\u4f1a", "\u5546\u4f1a", "\u8054\u5408\u4f1a", "\u59d4\u5458\u4f1a", "\u4fc3\u8fdb\u4f1a", "\u5b66\u4f1a",
    "Inc.", "Corp.", "Corporation", "LLC", "Ltd.", "Limited",
    "PLC", "GmbH", "AG", "SA", "S.A.", "Co.", "Group",
    "University", "College", "Institute", "School", "Laboratory", "Lab",
    "Bank", "Corp", "Incorporated",
]

_PERSON_SUFFIXES = [
    "\u5148\u751f", "\u5973\u58eb", "\u540c\u5fd7", "\u6559\u6388", "\u535a\u58eb", "\u533b\u751f", "\u8001\u5e08", "\u5f8b\u5e08",
    "\u4e3b\u5e2d", "\u603b\u7edf", "\u603b\u7406", "\u90e8\u957f", "\u4e3b\u4efb", "\u4e66\u8bb0", "\u5c40\u957f", "\u9662\u957f",
    "\u6821\u957f", "\u4f1a\u957f", "\u8463\u4e8b\u957f", "\u603b\u7ecf\u7406", "\u603b\u88c1", "\u603b\u76d1", "\u7ecf\u7406",
    "\u8bb0\u8005", "\u7f16\u8f91", "\u4f5c\u8005", "\u5206\u6790\u5e08", "\u4e13\u5bb6",
    "\u5148\u751f/\u5973\u58eb", "\u540c\u5fd7/\u5973\u58eb",
    "Jr.", "Sr.", "II", "III", "IV",
]


@dataclass
class EntityResult:
    entity_type: str
    extracted_text: str        # Qwen extraction result, "NONE" or "NO_RETRIEVED_ENTITY"
    original_entities: List[str] = field(default_factory=list)  # entities from original text
    normalized_entities: List[str] = field(default_factory=list) # normalized entities
    is_type_missing: bool = False   # True if original text had no entities of this type
    search_urls: List[str] = field(default_factory=list)
    fetched_texts: List[str] = field(default_factory=list)
    source_doc_index: int = 0      # which search result the extraction came from
    evidence_span: str = ""        # evidence text snippet
    success: bool = False


class Module2EntityExtractor:
    """Multi-threaded entity extraction for Module 2.

    Architecture per \u00a79.2.3:
      1. spaCy NER once (single thread)
      2. ThreadPoolExecutor(max_workers=7) \u2014 one per entity type.
         Each thread: mask \u2192 search \u2192 fetch \u2192 Qwen extraction.

    Three states per \u00a74.2.5:
      - TYPE MISSING: original text has no entities of this type \u2192 absent_embedding
      - RETRIEVAL EMPTY: entities exist but search returned nothing
      - NORMAL COMPARISON: entities exist and search returned results
    """

    # SpaCy label to entity type mapping (\u00a74.2.1)
    SPACY_LABEL_MAP = {
        "DATE": "time", "TIME": "time",
        "GPE": "location", "LOC": "location",
        "PERSON": "person", "PER": "person",
        "ORG": "organization",
        "CARDINAL": "number", "MONEY": "number", "PERCENT": "number",
        "EVENT": "event", "LAW": "event",
        "WORK_OF_ART": "object",
    }

    def __init__(
        self,
        api_workers: int = 7,
        search_top_k: int = 5,
        fetch_max_urls: int = 5,
        baidu_client: Optional[BaiduSearchClient] = None,
        qwen_client: Optional[QwenClient] = None,
        web_fetcher: Optional[WebFetcher] = None,
    ):
        self.api_workers = min(api_workers, len(ENTITY_TYPES))
        self.search_top_k = search_top_k
        self.fetch_max_urls = fetch_max_urls

        # Shared clients (thread-safe for concurrent HTTP)
        self.baidu = baidu_client or BaiduSearchClient()
        self.qwen = qwen_client or QwenClient()
        self.fetcher = web_fetcher or WebFetcher()

        # SpaCy model cache (lazy loaded, one per language)
        self._nlp_cache: Dict[str, 'spacy.Language'] = {}

    # -----------------------------------------------------------------------
    # Step \u2460: spaCy NER on original text (\u00a74.2.1)
    # -----------------------------------------------------------------------

    def _extract_entities_spacy(self, text: str, lang: str) -> Dict[str, List[Dict]]:
        """Extract entities from original text using spaCy.

        Returns dict mapping entity_type \u2192 list of {text, start, end} spans.
        SpaCy model is cached after first load per language.
        """
        import spacy
        model_name = "zh_core_web_sm" if lang == "zh" else "en_core_web_sm"

        # Cache hit \u2192 reuse loaded model
        if model_name in self._nlp_cache:
            nlp = self._nlp_cache[model_name]
        else:
            try:
                nlp = spacy.load(model_name)
                self._nlp_cache[model_name] = nlp
            except OSError:
                # spaCy model not available \u2192 try regex fallback
                return self._extract_entities_regex(text, lang)

        doc = nlp(text)
        entities = {t: [] for t in ENTITY_TYPES}
        for ent in doc.ents:
            mapped = self.SPACY_LABEL_MAP.get(ent.label_, "")
            if mapped:
                entry = {
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                # Avoid duplicate identical spans
                if entry not in entities[mapped]:
                    entities[mapped].append(entry)
        return entities

    def _extract_entities_regex(self, text: str, lang: str) -> Dict[str, List[Dict]]:
        """Regex fallback entity extraction when spaCy model unavailable."""
        entities = {t: [] for t in ENTITY_TYPES}

        if lang == "zh":
            # Time: Chinese date patterns
            for m in re.finditer(r'\d{4}\u5e74\d{1,2}\u6708\d{1,2}\u65e5', text):
                entities["time"].append({"text": m.group(), "start": m.start(), "end": m.end()})
            for m in re.finditer(r'\d{4}\u5e74\d{1,2}\u6708', text):
                entities["time"].append({"text": m.group(), "start": m.start(), "end": m.end()})

            # Number: digits + units
            for m in re.finditer(r'\d+[\u4e07\u4ebf]?\s*[\u5143%]', text):
                entities["number"].append({"text": m.group(), "start": m.start(), "end": m.end()})

            # Location: common suffixes
            for m in re.finditer(r'[\u4e00-\u9fff]{2,}(?:\u7701|\u5e02|\u533a|\u53bf|\u9547|\u6751|\u8857|\u8def)', text):
                entities["location"].append({"text": m.group(), "start": m.start(), "end": m.end()})

            # Organization: common suffixes
            for m in re.finditer(r'[\u4e00-\u9fff]{2,}(?:\u5927\u5b66|\u5b66\u9662|\u516c\u53f8|\u96c6\u56e2|\u94f6\u884c|\u5c40|\u90e8|\u59d4|\u4e2d\u5fc3|\u793e)', text):
                entities["organization"].append({"text": m.group(), "start": m.start(), "end": m.end()})

            # Person: names followed by titles
            for m in re.finditer(r'[\u4e00-\u9fff]{2,4}(?:\u5148\u751f|\u5973\u58eb|\u540c\u5fd7|\u6559\u6388|\u4e3b\u5e2d|\u603b\u7edf|\u603b\u7406|\u90e8\u957f|\u4e3b\u4efb|\u8bb0\u8005)', text):
                entities["person"].append({"text": m.group(), "start": m.start(), "end": m.end()})
        else:
            # English: number patterns
            for m in re.finditer(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', text):
                entities["number"].append({"text": m.group(), "start": m.start(), "end": m.end()})
            # Date patterns
            for m in re.finditer(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text):
                entities["time"].append({"text": m.group(), "start": m.start(), "end": m.end()})

        return entities

    # -----------------------------------------------------------------------
    # Step \u2461: Entity masking for search query (\u00a74.2.2)
    # -----------------------------------------------------------------------

    def _mask_entities_in_text(self, text: str, entities: List[Dict],
                                entity_type: str) -> str:
        """Mask entities of a specific type in text with [MASK_{TYPE}].

        Replaces by start_char descending to preserve offsets.
        """
        if not entities:
            return text

        mask_token = f"[MASK_{entity_type.upper()}]"
        sorted_ents = sorted(entities, key=lambda e: e["start"], reverse=True)

        masked = text
        for ent in sorted_ents:
            masked = masked[:ent["start"]] + mask_token + masked[ent["end"]:]

        return masked

    def _filter_blocked_terms(self, query: str) -> str:
        """Remove blocked terms from query to prevent label leakage (\u00a74.2.2)."""
        for term in BLOCKED_QUERY_TERMS:
            query = query.replace(term, "").replace(term.capitalize(), "")
        # Collapse multiple spaces
        query = re.sub(r'\s+', ' ', query).strip()
        return query if query else ""

    # -----------------------------------------------------------------------
    # Entity normalization (\u00a74.2.1)
    # -----------------------------------------------------------------------

    def _normalize_entity_text(self, text: str, entity_type: str,
                                lang: str) -> str:
        """Normalize entity text for consistent comparison across all 7 types (\u00a74.2.1).

        Applies type-specific normalization:
          - time: various date formats \u2192 YYYY-MM-DD
          - number: unit conversion (\u4e07/\u4ebf/million/billion \u2192 raw numbers)
          - location: alias expansion (\u6caa\u2192\u4e0a\u6d77, NYC\u2192New York City)
          - organization: remove redundant suffixes
          - person: remove honorific suffixes
          - event/object: case folding and whitespace normalization
        """
        if not text:
            return text

        text = text.strip()

        if entity_type == "time":
            return self._normalize_time(text, lang)

        if entity_type == "number":
            return self._normalize_number(text, lang)

        if entity_type == "location":
            return self._normalize_location(text, lang)

        if entity_type == "organization":
            return self._normalize_organization(text, lang)

        if entity_type == "person":
            return self._normalize_person(text, lang)

        # event / object: basic cleaning
        return text.strip()

    _DATE_PATTERNS_ZH = [
        (re.compile(r'(\d{4})\s*\u5e74\s*(\d{1,2})\s*\u6708\s*(\d{1,2})\s*\u65e5'), r'\1-\2-\3'),
        (re.compile(r'(\d{4})\s*\u5e74\s*(\d{1,2})\s*\u6708'), r'\1-\2'),
        (re.compile(r'(\d{1,2})\s*\u6708\s*(\d{1,2})\s*\u65e5'), r'??-\1-\2'),
        (re.compile(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})'), r'\1-\2-\3'),
        (re.compile(r'(\d{4})\u5e74(\d{1,2})\u6708(\d{1,2})'), r'\1-\2-\3'),
    ]

    _DATE_PATTERNS_EN = [
        # "Jan 1, 2024" or "January 1, 2024"
        (re.compile(r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                    r'\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', re.I),
         lambda m: self._month_to_number(m.group(1), m.group(2), m.group(3))),
        # "1 Jan 2024" or "1 January 2024"
        (re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+'
                    r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                    r',?\s+(\d{4})', re.I),
         lambda m: self._month_to_number(m.group(2), m.group(1), m.group(3))),
        (re.compile(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})'), r'\1-\2-\3'),
        (re.compile(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})'), r'\3-\1-\2'),
    ]

    _MONTH_MAP = {
        "jan": "01", "january": "01", "feb": "02", "february": "02",
        "mar": "03", "march": "03", "apr": "04", "april": "04",
        "may": "05", "jun": "06", "june": "06",
        "jul": "07", "july": "07", "aug": "08", "august": "08",
        "sep": "09", "september": "09", "oct": "10", "october": "10",
        "nov": "11", "november": "11", "dec": "12", "december": "12",
    }

    @classmethod
    def _month_to_number(cls, month_str: str, day_str: str, year_str: str) -> str:
        """Convert month name to number: 'January', 'Jan' -> '01'."""
        m = cls._MONTH_MAP.get(month_str.lower().strip(), "??")
        d = day_str.strip().zfill(2)
        return f"{year_str.strip()}-{m}-{d}"

    def _normalize_time(self, text: str, lang: str) -> str:
        """Unify date formats to YYYY-MM-DD."""
        original = text

        if lang == "zh":
            for pattern, replacement in self._DATE_PATTERNS_ZH:
                try:
                    text = pattern.sub(replacement, text)
                except Exception:
                    continue
        else:
            for pattern, replacement in self._DATE_PATTERNS_EN:
                try:
                    if callable(replacement):
                        text = pattern.sub(replacement, text)
                    else:
                        text = pattern.sub(replacement, text)
                except Exception:
                    continue

        # If no pattern matched, try generic date digit normalization
        if text == original:
            # Just pad single-digit months/days in any remaining YYYY-M-D patterns
            text = re.sub(r'(\d{4})-(\d)(?!\d)', r'\1-0\2', text)
            text = re.sub(r'(\d{4}-\d{2})-(\d)(?!\d)', r'\1-0\2', text)

        return text

    def _normalize_number(self, text: str, lang: str) -> str:
        """Unify number units: \u4e07\u219210000, \u4ebf\u2192100000000, million\u21921000000, etc."""
        if lang == "zh":
            # "2.5\u4e07" \u2192 "25000"
            text = re.sub(
                r'(\d+(?:\.\d+)?)\s*\u4e07',
                lambda m: str(int(float(m.group(1)) * 10000)), text
            )
            # "2.5\u4ebf" \u2192 "250000000"
            text = re.sub(
                r'(\d+(?:\.\d+)?)\s*\u4ebf',
                lambda m: str(int(float(m.group(1)) * 100000000)), text
            )
            # "1\u4e07" variant without decimal
            text = re.sub(
                r'(\d+)\s*\u4e07\u4ebf',
                lambda m: str(int(m.group(1)) * 1000000000000), text
            )
        else:
            # "5 million" \u2192 "5000000"
            text = re.sub(
                r'(\d+(?:\.\d+)?)\s*(?:million|M)(?:\b|$)',
                lambda m: str(int(float(m.group(1)) * 1000000)), text,
                flags=re.I
            )
            # "3 billion" \u2192 "3000000000"
            text = re.sub(
                r'(\d+(?:\.\d+)?)\s*(?:billion|B)(?:\b|$)',
                lambda m: str(int(float(m.group(1)) * 1000000000)), text,
                flags=re.I
            )
            # "2 trillion" \u2192 "2000000000000"
            text = re.sub(
                r'(\d+(?:\.\d+)?)\s*(?:trillion|T)(?:\b|$)',
                lambda m: str(int(float(m.group(1)) * 1000000000000)), text,
                flags=re.I
            )
            # "5 thousand" \u2192 "5000"
            text = re.sub(
                r'(\d+(?:\.\d+)?)\s*(?:thousand|K)(?:\b|$)',
                lambda m: str(int(float(m.group(1)) * 1000)), text,
                flags=re.I
            )
            # Remove commas in numbers: "1,234,567" \u2192 "1234567"
            text = re.sub(r'(?<=\d),(?=\d)', '', text)
            # "$50" prefix
            text = re.sub(r'^\$\s*', '', text)

        return text

    def _normalize_location(self, text: str, lang: str) -> str:
        """Expand location abbreviations to full names."""
        # Check full alias map first
        if text in _LOCATION_ALIASES:
            return _LOCATION_ALIASES[text]

        # For Chinese, check if text ends with a known short-form province/city
        if lang == "zh" and len(text) <= 2:
            if text in _LOCATION_ALIASES:
                return _LOCATION_ALIASES[text]

        return text

    def _normalize_organization(self, text: str, lang: str) -> str:
        """Remove redundant suffixes from organization names."""
        # Iteratively remove longest matching suffix
        for suffix in sorted(_ORG_SUFFIXES, key=len, reverse=True):
            if text.endswith(suffix) and len(text) > len(suffix):
                # Don't strip if the remaining part is too short (e.g., "\u5317\u4eac\u5927\u5b66" \u2192 strip "\u5927\u5b66" \u2192 "\u5317\u4eac" is OK but "\u5317\u5927" alone is not)
                remaining = text[:-len(suffix)].strip()
                if len(remaining) >= 2:
                    text = remaining
                    break  # Only remove one level of suffix
        return text.strip()

    def _normalize_person(self, text: str, lang: str) -> str:
        """Remove honorifics and titles from person names."""
        for suffix in sorted(_PERSON_SUFFIXES, key=len, reverse=True):
            if text.endswith(suffix) and len(text) > len(suffix):
                remaining = text[:-len(suffix)].strip()
                if len(remaining) >= 2:
                    text = remaining
                    break
        return text.strip()

    # -----------------------------------------------------------------------
    # Per-type processing (runs in parallel threads)
    # -----------------------------------------------------------------------

    def _process_type(
        self,
        entity_type: str,
        news_text: str,
        lang: str,
        original_entities: List[Dict],
    ) -> EntityResult:
        """Process one entity type: mask \u2192 search \u2192 fetch \u2192 extract.

        Called inside ThreadPoolExecutor. Each thread processes one type.
        """
        try:
            # Extract entity texts for branch text
            entity_texts = [e["text"] for e in original_entities]
            normalized_entities = [
                self._normalize_entity_text(e, entity_type, lang)
                for e in entity_texts
            ]

            if not original_entities:
                # TYPE MISSING (\u00a74.2.5): no entities of this type in text
                # \u2192 use absent_embedding in Stage 2 training
                return EntityResult(
                    entity_type=entity_type,
                    extracted_text="NONE",
                    original_entities=[],
                    normalized_entities=[],
                    is_type_missing=True,
                    success=True,
                )

            # Build search query: mask entities of this type \u2192 filter blocked terms
            masked_text = self._mask_entities_in_text(
                news_text, original_entities, entity_type
            )
            query = self._filter_blocked_terms(masked_text)
            if not query:
                return EntityResult(
                    entity_type=entity_type,
                    extracted_text="NONE",
                    original_entities=entity_texts,
                    normalized_entities=normalized_entities,
                    success=True,
                )

            # Baidu search
            search_results = self.baidu.search(query, top_k=self.search_top_k)
            if not search_results:
                # RETRIEVAL EMPTY (\u00a74.2.5): entities exist but no search results
                return EntityResult(
                    entity_type=entity_type,
                    extracted_text="NO_RETRIEVED_ENTITY",
                    original_entities=entity_texts,
                    normalized_entities=normalized_entities,
                    success=True,
                )

            # HTTP fetch full text from URLs
            urls = [r.url for r in search_results if r.url]
            snippets = [r.snippet for r in search_results if r.snippet]
            fetched_texts = self.fetcher.fetch_batch(urls, max_urls=self.fetch_max_urls)

            # Combine: full-text pages first, then snippets
            context_texts = list(fetched_texts)
            snippet_texts = [s for s in snippets if len(s) > 50]
            context_texts.extend(snippet_texts)

            if not context_texts:
                return EntityResult(
                    entity_type=entity_type,
                    extracted_text="NO_RETRIEVED_ENTITY",
                    original_entities=entity_texts,
                    normalized_entities=normalized_entities,
                    search_urls=urls,
                    success=True,
                )

            # Qwen extraction from top sources
            combined = "\n\n".join(context_texts[:3])
            extracted = ""
            try:
                if lang == "zh":
                    extracted = self.qwen.extract_entities(combined, entity_type,
                                                            context=news_text)
                else:
                    extracted = self.qwen.extract_entities_en(combined, entity_type,
                                                               context=news_text)
            except Exception:
                extracted = ""

            if not extracted:
                # Fallback: local spaCy/regex extraction on search results (\u00a74.2.4)
                try:
                    local_entities = self._extract_entities_spacy(combined, lang)
                    type_entities = local_entities.get(entity_type, [])
                    if type_entities:
                        extracted = "; ".join([e["text"] for e in type_entities[:5]])
                except Exception:
                    extracted = ""
            if not extracted:
                extracted = "NONE"
            extracted_text = extracted

            return EntityResult(
                entity_type=entity_type,
                extracted_text=extracted_text,
                original_entities=entity_texts,
                normalized_entities=normalized_entities,
                search_urls=urls,
                fetched_texts=fetched_texts,
                source_doc_index=0,
                evidence_span=context_texts[0][:200] if context_texts else "",
                success=True,
            )

        except Exception as e:
            return EntityResult(
                entity_type=entity_type,
                extracted_text="NONE",
                original_entities=[e["text"] for e in original_entities]
                if original_entities else [],
                normalized_entities=[],
                success=False,
            )

    # -----------------------------------------------------------------------
    # Main extraction API
    # -----------------------------------------------------------------------

    def extract_all(
        self, news_text: str, lang: str
    ) -> Dict[str, EntityResult]:
        """Extract all 7 entity types.

        Steps:
          1. spaCy NER once on original text (single thread)
          2. ThreadPoolExecutor(7) for per-type search + Qwen extraction

        Args:
            news_text: original news text
            lang: "zh" or "en"

        Returns:
            Dict mapping entity_type \u2192 EntityResult
        """
        # Step 1: spaCy NER on original text (single thread)
        all_entities = self._extract_entities_spacy(news_text, lang)

        # Step 2: Parallel per-type processing (\u00a79.2.3)
        results = {}
        with ThreadPoolExecutor(max_workers=self.api_workers) as executor:
            futures = {
                executor.submit(
                    self._process_type, etype, news_text, lang,
                    all_entities.get(etype, [])
                ): etype
                for etype in ENTITY_TYPES
            }
            for future in as_completed(futures):
                etype = futures[future]
                try:
                    results[etype] = future.result()
                except Exception:
                    results[etype] = EntityResult(
                        entity_type=etype,
                        extracted_text="NONE",
                        success=False,
                    )

        return {t: results.get(t, EntityResult(
            entity_type=t, extracted_text="NONE", success=False
        )) for t in ENTITY_TYPES}

    # -----------------------------------------------------------------------
    # Entity statistics for ablation baseline (\u00a74.2.7)
    # -----------------------------------------------------------------------

    def compute_entity_stats(
        self, results: Dict[str, EntityResult]
    ) -> np.ndarray:
        """Compute 35-dim entity statistics for ablation experiments.

        5 statistics \u00d7 7 entity types:
          - num_original: count of original entities (capped at 5)
          - num_retrieved: count of retrieved entities (capped at 10)
          - overlap_rate: token-level overlap ratio [0, 1]
          - conflict_proxy: -1=no retrieval, 0=has overlap, 1=completely different
          - has_entity: 0/1 whether original text has this entity type

        These statistics encode similar information to the 7 BERT CLS branches
        but in a coarser form, useful for ablation baselines (\u00a74.2.7).

        Returns:
            (35,) numpy array
        """
        stats_list = []
        for etype in ENTITY_TYPES:
            result = results.get(etype)

            if not result or result.is_type_missing:
                stats_list.extend([0.0, 0.0, 0.0, -1.0, 0.0])
                continue

            num_orig = min(len(result.original_entities), 5)

            # Parse retrieved entities from extracted_text
            retrieved_text = result.extracted_text or ""
            if retrieved_text in ("NONE", "NO_RETRIEVED_ENTITY", ""):
                num_ret = 0
                overlap = 0.0
                conflict = -1.0
            else:
                # Count entities from Qwen output (split by common delimiters)
                parts = [p.strip() for p in re.split(r'[;,\uff0c\uff1b]', retrieved_text) if p.strip()]
                num_ret = min(len(parts), 10)

                # Token-level overlap
                orig_tokens = set()
                for e in result.original_entities:
                    orig_tokens.update(e.lower().split())
                ret_tokens = set()
                for p in parts:
                    ret_tokens.update(p.lower().split())

                if not orig_tokens or not ret_tokens:
                    overlap = 0.0
                    conflict = 1.0
                else:
                    overlap = len(orig_tokens & ret_tokens) / max(len(orig_tokens | ret_tokens), 1)
                    conflict = 0.0 if overlap > 0 else 1.0

            stats_list.extend([float(num_orig), float(num_ret), overlap, conflict, 1.0])

        return np.array(stats_list, dtype=np.float32)

    def build_branch_texts(
        self, results: Dict[str, EntityResult], lang: str = "zh"
    ) -> Tuple[List[str], List[int]]:
        """Build Module 2 branch texts from extraction results.

        Format per \u00a74.2.5:
          "original_entities: \u7c7b\u578b:\u5b9e\u4f53 [SEP] retrieved_entities: \u7c7b\u578b:\u5b9e\u4f53 doc:\u7d22\u5f15 evidence:\u8bc1\u636e\u7247\u6bb5"

        Uses normalized entities (\u00a74.2.1) for consistent comparison between
        original and retrieved entities.

        Three states:
          - TYPE MISSING: is_type_missing=True \u2192 empty text, mask=0 (\u2192 absent_embedding)
          - RETRIEVAL EMPTY: extracted_text="NO_RETRIEVED_ENTITY" \u2192 mask=1, no doc/evidence
          - NORMAL: has extraction \u2192 full format with doc: and evidence:

        Args:
            results: entity extraction results dict
            lang: language of the source text, used for retrieved entity normalization

        Returns:
            (branch_texts: List[str], branch_mask: List[int])
            - branch_texts: 7 strings
            - branch_mask: 7 ints, 1 if branch has valid text (\u2260 absent_embedding)
        """
        branch_texts = []
        branch_mask = []

        for etype in ENTITY_TYPES:
            result = results.get(etype)

            if not result or result.is_type_missing:
                # TYPE MISSING: empty text, mask=0 \u2192 Stage 2 uses absent_embedding
                branch_texts.append("")
                branch_mask.append(0)
                continue

            # Use normalized entities for fair comparison with retrieved entities
            orig_entities_for_text = (
                result.normalized_entities
                if result.normalized_entities
                else result.original_entities
            )
            orig_entities_str = "; ".join(orig_entities_for_text[:5])

            if result.extracted_text in ("NO_RETRIEVED_ENTITY", "", None):
                # RETRIEVAL EMPTY: entities exist but no search results
                text = (
                    f"original_entities: {etype}: {orig_entities_str}"
                    f" [SEP] retrieved_entities: {etype}: NO_RETRIEVED_ENTITY"
                )
                branch_texts.append(text)
                branch_mask.append(1)
            elif result.extracted_text == "NONE":
                # Qwen returned no relevant extraction
                text = (
                    f"original_entities: {etype}: {orig_entities_str}"
                    f" [SEP] retrieved_entities: {etype}: NONE"
                )
                branch_texts.append(text)
                branch_mask.append(1)
            else:
                # NORMAL: valid extraction with doc and evidence
                # Normalize retrieved entities for consistent comparison
                retrieved_text = result.extracted_text or ""
                retrieved_normalized = self._normalize_retrieved_text(
                    retrieved_text, etype, lang
                )

                doc_str = f"doc:{result.source_doc_index}"
                ev_str = f"evidence:{result.evidence_span[:100]}" if result.evidence_span else ""
                extra_parts = [doc_str, ev_str] if ev_str else [doc_str]

                text = (
                    f"original_entities: {etype}: {orig_entities_str}"
                    f" [SEP] retrieved_entities: {etype}: {retrieved_normalized[:300]}"
                    f" {' '.join(extra_parts)}"
                )
                branch_texts.append(text)
                branch_mask.append(1)

        return branch_texts, branch_mask

    def _normalize_retrieved_text(self, text: str, entity_type: str, lang: str = "zh") -> str:
        """Normalize each entity in a Qwen-retrieved text string.

        Qwen output is a delimited string like "\u5f20\u4e09 ; \u674e\u56db ; \u738b\u4e94".
        Splits on common delimiters, normalizes each, rejoins with "; ".
        Falls back to raw text if splitting fails.

        Args:
            text: Qwen-extracted entity text
            entity_type: entity type for normalization rules
            lang: language of the source text ("zh" or "en")
        """
        if not text or text in ("NONE", "NO_RETRIEVED_ENTITY"):
            return text
        parts = [p.strip() for p in re.split(r'[;,\uff0c\uff1b]', text) if p.strip()]
        if not parts:
            return text
        normalized = []
        for part in parts:
            norm = self._normalize_entity_text(part, entity_type, lang)
            normalized.append(norm)
        return "; ".join(normalized)
