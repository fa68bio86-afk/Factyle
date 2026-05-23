"""Full-text web page fetcher with aggressive content cleaning.

Reference: ARCHITECTURE.md §4.2.3, §9.2.3

Fetches full text from search result URLs and performs aggressive cleaning:
  - HTML tags, scripts, styles
  - Navigation/menus, ads, sidebars
  - Comment sections, footers
  - Irrelevant link lists

Parameters per §4.2.3:
  - max_chars: 6000 per document
  - min_chars: 300 (fall back to snippet if below)
  - timeout: 12s per request
"""

import re
from typing import List, Optional

import requests


class WebFetcher:
    """Fetch and aggressively clean web page content.

    Used by Module 2 to get full text from entity search results.
    Designed for multi-threaded use (one fetcher instance per thread is safe).
    """

    # Aggressive cleaning patterns: sections to remove entirely (§4.2.3)
    REMOVE_SECTION_PATTERNS = [
        # Chinese ad/nav/comment blocks (广告=ad, 推广=promoted, 分享=share, etc.)
        r'<div[^>]*?(?:ad|banner|推广|广告|recommend|related|footer|header|nav|menu|sidebar|comment|分享)[^>]*>.*?</div>',
        r'<ul[^>]*?(?:nav|menu|ad|related|comment)[^>]*>.*?</ul>',
        # Common ad/widget containers
        r'<iframe[^>]*>.*?</iframe>',
        r'<ins[^>]*>.*?</ins>',
        # Navigation with text markers
        r'<(?:nav|footer|aside)[^>]*>.*?</(?:nav|footer|aside)>',
        # Comment sections
        r'<section[^>]*?(?:comment|discussion|feedback)[^>]*>.*?</section>',
    ]

    def __init__(
        self,
        timeout: int = 12,
        max_chars: int = 6000,
        min_chars: int = 300,
        user_agent: Optional[str] = None,
    ):
        self.timeout = timeout
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    def fetch(self, url: str) -> Optional[str]:
        """Fetch and aggressively clean full text from a URL.

        Args:
            url: target webpage URL

        Returns:
            Cleaned text content, or None if fetch fails or content is too short
        """
        if not url or not url.startswith("http"):
            return None

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
                allow_redirects=True,
            )
            resp.raise_for_status()

            # Detect encoding
            resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
            html = resp.text

            text = self._clean_html(html)

            # Check minimum content length per §4.2.3 (< 300 chars → None)
            if len(text) < self.min_chars:
                return None

            return text[:self.max_chars]

        except requests.RequestException:
            return None

    def _clean_html(self, html: str) -> str:
        """Aggressive HTML cleaning per §4.2.3.

        Removal order:
          1. Scripts, styles, noscript, iframes
          2. Ad/nav/comment containers (regex patterns)
          3. All HTML tags
          4. HTML entities
          5. Ad/nav/comment text patterns
          6. Extra whitespace
        """
        text = html

        # Step 1: Remove script, style, noscript, iframe blocks
        for tag in ['script', 'style', 'noscript', 'iframe', 'svg', 'canvas']:
            text = re.sub(
                rf'<{tag}[^>]*>.*?</{tag}>', '',
                text, flags=re.DOTALL | re.IGNORECASE
            )

        # Step 2: Remove ad/nav/comment containers
        for pattern in self.REMOVE_SECTION_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # Step 3: Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Step 4: Replace block tags with newlines
        text = re.sub(
            r'</?(?:div|p|br|li|tr|h[1-6]|section|article|blockquote|pre|code|th|td)[^>]*>',
            '\n', text, flags=re.IGNORECASE
        )

        # Step 5: Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Step 6: Decode common HTML entities
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        text = re.sub(r'&#\d+;', ' ', text)

        # Step 7: Remove ad/nav/comment text patterns
        ad_patterns = [
            # Chinese boilerplate: 导航=nav, 广告=ad, 评论=comment, 分享=share, etc.
            r'(?:导航|菜单|广告|推广|相关推荐|评论|登录|注册|分享|点赞|关注|订阅'
            r'|免责声明|版权|转载|来源|责任编辑|编辑|作者|校对|排版|美编)'
            r'\s*[:：]?[^。\n]{0,50}',
            # English boilerplate
            r'(?:nav|menu|advertisement|ad\s*|related\s*|comment|login|register'
            r'|share|subscribe|follow|disclaimer|copyright|published|updated|tags?'
            r'|categories?|leave a reply|click here|read more)'
            r'\s*[:：]?[^。\n]{0,50}',
            # URL-only lines
            r'^https?://\S+$',
            # Short lines (navigation/text links)
            r'^\s*\w{1,30}\s*$',
        ]
        for pattern in ad_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.MULTILINE)

        # Step 8: Collapse whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def fetch_batch(
        self, urls: list, max_urls: int = 5
    ) -> List[str]:
        """Fetch multiple URLs, return valid texts.

        Used in Module 2 entity pipeline: fetch up to 5 search result URLs.
        """
        results = []
        for url in urls[:max_urls]:
            text = self.fetch(url)
            if text and len(text) >= self.min_chars:
                results.append(text)
        return results
