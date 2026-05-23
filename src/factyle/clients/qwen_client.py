"""Qwen (通义千问) LLM API client.

Reference: ARCHITECTURE.md §4.2.4, §5.2, §9.2.3-4

Uses OpenAI-compatible API provided by DashScope:
  https://dashscope.aliyuncs.com/compatible-mode/v1

Two model sizes:
  - Qwen3-8B: entity extraction (Module 2), style rewriting (Module 3)
  - Qwen3-32B: style analysis (Module 3)
"""

import json
import os
import time
from typing import List, Optional

import requests


class QwenClient:
    """Client for Qwen LLM via DashScope OpenAI-compatible API.

    Supports multi-threaded concurrent calls (not safe for GPU use).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_8b: str = "qwen3-8b",
        model_32b: str = "qwen3-32b",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.api_base = (api_base or os.environ.get("QWEN_API_BASE", "")
                         or "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model_8b = model_8b
        self.model_32b = model_32b
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("QWEN_API_KEY not set. Check .env file.")

    def _chat(
        self,
        messages: List[dict],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Call Qwen chat completion API.

        Args:
            messages: OpenAI-format message list
            model: model name string
            temperature: sampling temperature (low = deterministic for extraction)
            max_tokens: max output tokens

        Returns:
            Response text content
        """
        url = f"{self.api_base.rstrip('/')}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "enable_thinking": False,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "")
                if choice.get("finish_reason") == "length":
                    pass  # truncated — acceptable for extraction tasks
                return content.strip()

            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Qwen API error (model={model}): {e}"
                    ) from e
                time.sleep(2.0 * (attempt + 1))

        return ""

    def extract_entities(
        self, text: str, entity_type: str, context: str = ""
    ) -> str:
        """Extract entity of given type from search context (Chinese).

        Prompts Qwen (in Chinese) to extract entities of a specific type
        from search result text, using the original news text as context.

        Args:
            text: search result text (full page or snippet)
            entity_type: type label (person, location, organization, ...)
            context: original news text for reference

        Returns:
            Extracted entity description or "NONE"
        """
        # Prompt: extract {entity_type} entities related to the news from search results
        prompt = (
            f"从以下搜索结果中提取与新闻相关的「{entity_type}」实体列表。\n\n"
            f"新闻原文（参考）：{context[:500]}\n\n"
            f"搜索结果：{text[:3000]}\n\n"
            "要求：\n"
            f"1. 只提取与新闻相关的{entity_type}实体名称，不要描述\n"
            "2. 多个实体用分号(;)分隔，如：实体1；实体2；实体3\n"
            "3. 如果搜索结果中不包含该类型实体的相关信息，只输出 NONE\n"
            "4. 不要编造信息"
        )
        return self._chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_8b,
            temperature=0.1,
        )

    def extract_entities_en(
        self, text: str, entity_type: str, context: str = ""
    ) -> str:
        """Extract entity of given type from search context (English)."""
        prompt = (
            f"Extract '{entity_type}' entities related to this news from the search results.\n\n"
            f"Original news (reference): {context[:500]}\n\n"
            f"Search results: {text[:3000]}\n\n"
            "Requirements:\n"
            f"1. Only extract {entity_type} entity names, do not describe\n"
            "2. Separate multiple entities with semicolons (;), e.g.: entity1; entity2; entity3\n"
            "3. If no relevant information for this entity type, output only NONE\n"
            "4. Do not fabricate information"
        )
        return self._chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_8b,
            temperature=0.1,
        )

    def rewrite_style(self, text: str, style: str, lang: str = "zh") -> str:
        """Rewrite news text in a given style (Module 3 Ta/Tb).

        Args:
            text: original news text
            style: "authoritative" (Ta) or "sensational" (Tb)
            lang: "zh" or "en"

        Returns:
            Rewritten text
        """
        if lang == "en":
            style_desc = {
                "authoritative": "formal, objective, citing official sources",
                "sensational": "exaggerated, emotional, clickbait-style",
            }.get(style, style)
            prompt = (
                f"Rewrite the following news in a {style_desc} style.\n"
                f"Keep the key facts unchanged. Only change the tone.\n\n"
                f"Original: {text[:2000]}\n\n"
                f"Rewritten ({style} style):"
            )
        else:
            # Chinese prompt: rewrite news in an authoritative or sensational style
            style_desc = {
                "authoritative": "权威、客观、引用官方来源",
                "sensational": "夸张、情绪化、标题党风格",
            }.get(style, style)
            prompt = (
                f"将以下新闻改写成{style_desc}风格。\n"
                f"保持关键事实不变，只改变语气。\n\n"
                f"原文：{text[:2000]}\n\n"
                f"改写（{style}风格）："
            )

        return self._chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_8b,
            temperature=0.3,
            max_tokens=2048,
        )

    def analyze_style(
        self,
        original_text: str,
        ta_skeleton: str,
        tb_skeleton: str,
        lang: str = "zh",
    ) -> str:
        """Analyze style differences between Ta and Tb skeletons.

        Args:
            original_text: original news text
            ta_skeleton: authoritative rewrite skeleton (entity-masked)
            tb_skeleton: sensational rewrite skeleton (entity-masked)
            lang: "zh" or "en"

        Returns:
            Style difference analysis text (input for BERT CLS)
        """
        if lang == "en":
            prompt = (
                "Compare the following three texts and analyze their style differences.\n"
                "Focus on tone, formality, emotional language, and narrative structure.\n"
                "Do NOT judge truthfulness or label as fake/real.\n\n"
                f"Original: {original_text[:1000]}\n\n"
                f"Authoritative rewrite skeleton: {ta_skeleton[:1000]}\n\n"
                f"Sensational rewrite skeleton: {tb_skeleton[:1000]}\n\n"
                "Style difference analysis:"
            )
        else:
            # Chinese prompt: compare three texts and analyze style differences
            prompt = (
                "比较以下三篇文本的风格差异。\n"
                "关注语气、正式程度、情感语言和叙事结构。\n"
                "不要判断真假标签。\n\n"
                f"原文：{original_text[:1000]}\n\n"
                f"权威改写骨架：{ta_skeleton[:1000]}\n\n"
                f"煽情改写骨架：{tb_skeleton[:1000]}\n\n"
                "风格差异分析："
            )

        return self._chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_32b,
            temperature=0.1,
            max_tokens=1024,
        )
