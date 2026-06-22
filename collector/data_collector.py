"""
============================================================
  MODULE 1 — collector/data_collector.py
  FIXED VERSION: 8 sources, proper document tracking
============================================================

  Sources:
    1.  RSS feeds           (company + news + tech)
    2.  Reddit              (community sentiment)
    3.  NewsAPI             (optional, free key)
    4.  NVIDIA IR page      (official press releases)
    5.  HackerNews          (tech community discussion)
    6.  GitHub Trending     (AI/ML project trends)
    7.  Yahoo Finance News  (financial headlines)
    8.  Wikipedia           (company + competitor context)

  BUG FIXED: documents count was == chunks count because
  pipeline.py was using db_stats["total_chunks"] as
  "total_documents" when docs list was empty (skip-collection path).
  Now pipeline.py properly tracks doc count separately.
"""

import feedparser
import requests
from datetime import datetime
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    RSS_FEEDS, REDDIT_SUBREDDITS, REDDIT_POST_LIMIT,
    NEWS_API_KEY, NEWS_QUERY, NEWS_MAX_RESULTS,
    COMPANY_NAME, MAX_DOCUMENTS
)


# ──────────────────────────────────────────
#  Helper: parse any date → "YYYY-MM-DD HH:MM"
# ──────────────────────────────────────────
def _parse_date(raw) -> str:
    """
    Converts any date format to clean "YYYY-MM-DD HH:MM".

    Handles:
      RSS  → "Sun, 21 Jun 2026 23:59:00 +0000"
      ISO  → "2026-06-21T23:59:00Z"
      Date → "2026-06-21"

    The old code used str(date)[:20] which cut off mid-word:
      "Sun, 21 Jun 2026 23:" ← broken
    This function always returns a complete, human-readable string.
    """
    if not raw:
        return datetime.now().strftime("%Y-%m-%d %H:%M")
    raw = str(raw).strip()

    # Try RFC 2822 — standard RSS date format
    try:
        return parsedate_to_datetime(raw).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    # Try ISO 8601 variants
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:19], fmt).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    # Fallback — use current time
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ──────────────────────────────────────────
#  Helper: scrape plain text from a URL
# ──────────────────────────────────────────
def _scrape_text(url: str, max_chars: int = 2000) -> str:
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception:
        return ""


# ──────────────────────────────────────────
#  Source 1: RSS Feeds
# ──────────────────────────────────────────
def collect_rss(feeds: list = RSS_FEEDS) -> list[dict]:
    """
    Parse RSS feeds. Source name is set per-feed (not generic "RSS").
    Hyperparameter: RSS_FEEDS in settings.py
    """
    documents = []

    # Map feed URL → readable source name
    feed_names = {
        "nvidianews.nvidia.com":   "NVIDIA News",
        "yahoo.com":               "Yahoo Finance",
        "TechCrunch":              "TechCrunch",
        "theverge.com":            "The Verge",
        "electronicdesign.com":    "Electronic Design",
        "wired.com":               "Wired",
        "arstechnica.com":         "Ars Technica",
        "tomshardware.com":        "Tom's Hardware",
    }

    for feed_url in feeds:
        # Pick a friendly name
        src_name = "RSS Feed"
        for key, name in feed_names.items():
            if key in feed_url:
                src_name = name
                break

        print(f"  [RSS] Fetching: {src_name}")
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                url     = entry.get("link", "")
                date    = _parse_date(entry.get("published", ""))

                content = summary if len(summary) > 150 else _scrape_text(url)

                if title and content:
                    documents.append({
                        "title":   title,
                        "content": content,
                        "source":  src_name,
                        "url":     url,
                        "date":    date,
                    })
        except Exception as e:
            print(f"  [RSS] Error on {feed_url}: {e}")

    print(f"  [RSS] Collected {len(documents)} documents")
    return documents


# ──────────────────────────────────────────
#  Source 2: Reddit
# ──────────────────────────────────────────
def collect_reddit(
    subreddits: list = REDDIT_SUBREDDITS,
    post_limit: int  = REDDIT_POST_LIMIT,
) -> list[dict]:
    """
    Fetch hot posts from Reddit without auth using the JSON API.

    FIX: Reddit blocks requests with generic User-Agent strings.
    Must use a browser-like UA and include proper Accept headers.
    Also adds retry logic and graceful 429/403 handling.
    """
    import time

    # Reddit requires a descriptive User-Agent — generic ones get 403/429
    HEADERS = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36",
        "Accept":          "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    documents = []

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit={post_limit}&raw_json=1"
        print(f"  [Reddit] Fetching r/{sub} ...")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)

            # Rate-limited — skip this subreddit gracefully
            if resp.status_code == 429:
                print(f"  [Reddit] r/{sub} rate-limited (429) — skipping")
                time.sleep(2)
                continue
            if resp.status_code == 403:
                print(f"  [Reddit] r/{sub} access denied (403) — private/banned subreddit")
                continue
            if resp.status_code != 200:
                print(f"  [Reddit] r/{sub} HTTP {resp.status_code} — skipping")
                continue

            data = resp.json()

            # Guard against unexpected response shape
            children = data.get("data", {}).get("children", [])
            if not children:
                print(f"  [Reddit] r/{sub} returned 0 posts")
                continue

            for post in children:
                d     = post.get("data", {})
                title = d.get("title", "").strip()
                if not title:
                    continue

                body    = d.get("selftext", "").strip()
                # Skip [removed] / [deleted] posts
                if body in ("[removed]", "[deleted]"):
                    body = ""

                score   = d.get("score", 0)
                num_com = d.get("num_comments", 0)
                url_p   = "https://www.reddit.com" + d.get("permalink", "")
                created = d.get("created_utc", 0)
                date    = _parse_date(
                    datetime.fromtimestamp(created).isoformat() if created else ""
                )

                # Rich content: title + body + engagement metrics
                content = title
                if body:
                    content += f". {body}"
                content += f" [Upvotes: {score}, Comments: {num_com}]"

                documents.append({
                    "title":   title,
                    "content": content[:1500],
                    "source":  f"Reddit/r/{sub}",
                    "url":     url_p,
                    "date":    date,
                    "score":   score,
                })

            print(f"  [Reddit] r/{sub}: {len([p for p in children if p.get('data',{}).get('title')])} posts")
            time.sleep(1)   # polite delay between subreddits

        except ValueError as e:
            print(f"  [Reddit] r/{sub} JSON parse error: {e} — Reddit may be blocking requests")
        except Exception as e:
            print(f"  [Reddit] r/{sub} error: {type(e).__name__}: {e}")

    print(f"  [Reddit] Collected {len(documents)} documents total")
    return documents


# ──────────────────────────────────────────
#  Source 3: NewsAPI (optional)
# ──────────────────────────────────────────
def collect_newsapi(
    api_key: str     = NEWS_API_KEY,
    query: str       = NEWS_QUERY,
    max_results: int = NEWS_MAX_RESULTS,
) -> list[dict]:
    """
    Fetch from newsapi.org. Free key at newsapi.org
    Hyperparameters: NEWS_API_KEY, NEWS_QUERY, NEWS_MAX_RESULTS
    """
    if not api_key:
        print("  [NewsAPI] No key set — skipping (add NEWS_API_KEY in settings.py)")
        return []

    documents = []
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&pageSize={max_results}&sortBy=publishedAt&apiKey={api_key}"
        )
        resp = requests.get(url, timeout=10)
        for a in resp.json().get("articles", []):
            title   = a.get("title", "")
            content = a.get("description", "") or a.get("content", "")
            documents.append({
                "title":   title,
                "content": f"{title}. {content}",
                "source":  a.get("source", {}).get("name", "NewsAPI"),
                "url":     a.get("url", ""),
                "date":    _parse_date(a.get("publishedAt", "")),
            })
    except Exception as e:
        print(f"  [NewsAPI] Error: {e}")

    print(f"  [NewsAPI] Collected {len(documents)} documents")
    return documents


# ──────────────────────────────────────────
#  Source 4: NVIDIA Official Pages
# ──────────────────────────────────────────
def collect_nvidia_official() -> list[dict]:
    """Scrape NVIDIA's investor relations and blog pages."""
    documents = []
    pages = [
        ("https://ir.nvidia.com/press-releases",          "NVIDIA IR"),
        ("https://www.nvidia.com/en-us/about-nvidia/",    "NVIDIA About"),
        ("https://blogs.nvidia.com/",                      "NVIDIA Blog"),
        ("https://nvidianews.nvidia.com/",                 "NVIDIA Newsroom"),
    ]
    for url, src in pages:
        print(f"  [NVIDIA Official] Fetching {src}")
        text = _scrape_text(url, max_chars=3000)
        if text:
            documents.append({
                "title":   f"{src} — {datetime.now().date()}",
                "content": text,
                "source":  src,
                "url":     url,
                "date":    str(datetime.now().date()),
            })
    print(f"  [NVIDIA Official] Collected {len(documents)} documents")
    return documents


# ──────────────────────────────────────────
#  Source 5: HackerNews
# ──────────────────────────────────────────
def collect_hackernews(query: str = "NVIDIA", max_results: int = 30) -> list[dict]:
    """
    Search HackerNews via Algolia API — no key needed.
    Great for tech community sentiment on NVIDIA products.
    Hyperparameter: max_results (default 30)
    """
    documents = []
    print(f"  [HackerNews] Searching for '{query}'...")
    try:
        url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage={max_results}"
        resp = requests.get(url, timeout=8)
        hits = resp.json().get("hits", [])

        for hit in hits:
            title   = hit.get("title", "")
            points  = hit.get("points", 0)
            comments= hit.get("num_comments", 0)
            hn_url  = f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"
            date    = _parse_date(hit.get("created_at", ""))

            # Include engagement metrics for sentiment weighting
            content = f"{title}. Points: {points}, Comments: {comments}."
            # Optionally scrape the linked article
            story_url = hit.get("url", "")
            if story_url:
                extra = _scrape_text(story_url, max_chars=800)
                if extra:
                    content += " " + extra

            if title:
                documents.append({
                    "title":   title,
                    "content": content[:1500],
                    "source":  "HackerNews",
                    "url":     hn_url,
                    "date":    date,
                    "score":   points,
                })
    except Exception as e:
        print(f"  [HackerNews] Error: {e}")

    print(f"  [HackerNews] Collected {len(documents)} documents")
    return documents


# ──────────────────────────────────────────
#  Source 6: GitHub Trending (AI/ML repos)
# ──────────────────────────────────────────
def collect_github_trending() -> list[dict]:
    """
    Scrape GitHub trending page for AI/CUDA/ML repos.
    Shows what developers are building with NVIDIA tech.
    """
    documents = []
    urls = [
        ("https://github.com/trending/python?since=weekly&spoken_language_code=en", "GitHub Trending Python"),
        ("https://github.com/trending/cuda?since=weekly",                           "GitHub Trending CUDA"),
    ]

    print("  [GitHub] Fetching trending repos...")
    for url, src in urls:
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "lxml")
            repos = soup.select("article.Box-row")[:15]

            for repo in repos:
                name_tag = repo.select_one("h2 a")
                desc_tag = repo.select_one("p")
                stars_tag= repo.select_one("span.d-inline-block.float-sm-right")

                if name_tag:
                    name  = name_tag.get_text(strip=True).replace("\n","").replace(" ","")
                    desc  = desc_tag.get_text(strip=True) if desc_tag else ""
                    stars = stars_tag.get_text(strip=True) if stars_tag else "?"

                    documents.append({
                        "title":   f"GitHub Trending: {name}",
                        "content": f"Repository {name} is trending on GitHub. {desc} Stars this week: {stars}.",
                        "source":  src,
                        "url":     f"https://github.com/{name}",
                        "date":    str(datetime.now().date()),
                    })
        except Exception as e:
            print(f"  [GitHub] Error on {url}: {e}")

    print(f"  [GitHub] Collected {len(documents)} documents")
    return documents


# ──────────────────────────────────────────
#  Source 7: Yahoo Finance News (no key)
# ──────────────────────────────────────────
def collect_yahoo_finance(ticker: str = "NVDA") -> list[dict]:
    """
    Scrape Yahoo Finance news for NVDA.
    No API key required — just web scraping.
    Hyperparameter: ticker symbol
    """
    documents = []
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    print(f"  [Yahoo Finance] Fetching {ticker} news...")
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        soup = BeautifulSoup(resp.text, "lxml")

        # Find news article links and titles
        articles = soup.select("h3 a, h2 a")[:20]
        for a in articles:
            title = a.get_text(strip=True)
            href  = a.get("href", "")
            if href.startswith("/"):
                href = f"https://finance.yahoo.com{href}"

            if title and len(title) > 10:
                content = _scrape_text(href, max_chars=1000) if href.startswith("http") else title
                documents.append({
                    "title":   title,
                    "content": f"{title}. {content}",
                    "source":  "Yahoo Finance",
                    "url":     href,
                    "date":    str(datetime.now().date()),
                })
    except Exception as e:
        print(f"  [Yahoo Finance] Error: {e}")

    print(f"  [Yahoo Finance] Collected {len(documents)} documents")
    return documents


# ──────────────────────────────────────────
#  Source 8: Wikipedia — Company Context
# ──────────────────────────────────────────
def collect_wikipedia(topics: list = None) -> list[dict]:
    """
    Fetch Wikipedia summaries for NVIDIA and key competitors.
    Provides stable factual context for the LLM.
    Hyperparameter: topics list
    """
    if topics is None:
        topics = ["NVIDIA", "AMD", "Intel", "CUDA", "Graphics processing unit",
                  "Artificial intelligence chip", "Jensen Huang"]

    documents = []
    print(f"  [Wikipedia] Fetching {len(topics)} articles...")
    for topic in topics:
        try:
            url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ','_')}"
            resp = requests.get(url, timeout=8, headers={"User-Agent": "nvidia-agent/1.0"})
            data = resp.json()

            title   = data.get("title", topic)
            extract = data.get("extract", "")
            wiki_url= data.get("content_urls", {}).get("desktop", {}).get("page", "")

            if extract:
                documents.append({
                    "title":   f"Wikipedia: {title}",
                    "content": extract,
                    "source":  "Wikipedia",
                    "url":     wiki_url,
                    "date":    str(datetime.now().date()),
                })
        except Exception as e:
            print(f"  [Wikipedia] Error on {topic}: {e}")

    print(f"  [Wikipedia] Collected {len(documents)} documents")
    return documents


# ──────────────────────────────────────────
#  Main Entry Point
# ──────────────────────────────────────────
def collect_all() -> list[dict]:
    """
    Run all 8 collectors and return deduplicated documents.
    Returns list of dicts — length = actual number of DOCUMENTS (not chunks).
    """
    print(f"\n{'='*50}")
    print(f"  Starting data collection for {COMPANY_NAME}")
    print(f"  Sources: RSS, Reddit, NewsAPI, NVIDIA IR, HackerNews,")
    print(f"           GitHub Trending, Yahoo Finance, Wikipedia")
    print(f"{'='*50}")

    all_docs = []
    all_docs += collect_rss()
    all_docs += collect_reddit()
    all_docs += collect_newsapi()
    all_docs += collect_nvidia_official()
    all_docs += collect_hackernews()
    all_docs += collect_github_trending()
    all_docs += collect_yahoo_finance()
    all_docs += collect_wikipedia()

    # Deduplicate by title (case-insensitive)
    seen_titles = set()
    unique_docs = []
    for doc in all_docs:
        t = doc["title"].lower().strip()[:100]
        if t not in seen_titles and len(doc.get("content","")) > 50:
            seen_titles.add(t)
            unique_docs.append(doc)

    unique_docs = unique_docs[:MAX_DOCUMENTS]

    # Print source breakdown
    from collections import Counter
    source_counts = Counter(d["source"] for d in unique_docs)
    print(f"\n  Source breakdown:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:30s} {cnt} docs")

    print(f"\n  Total unique documents: {len(unique_docs)}")
    return unique_docs


if __name__ == "__main__":
    docs = collect_all()
    print(f"\nSample: {docs[0] if docs else 'None'}")
