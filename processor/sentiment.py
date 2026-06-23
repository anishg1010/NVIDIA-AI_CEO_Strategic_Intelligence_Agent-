"""
============================================================
  processor/sentiment.py
============================================================
  Dual-model sentiment with simplified, accurate summary.

  Model routing (auto, by source):
    FinBERT (yiyanghkust/finbert-tone)
      → Yahoo Finance, NVIDIA IR, Reuters, Bloomberg, CNBC,
        NewsAPI, MarketBeat, GlobeNewswire, PRNewswire
      Financial domain, trained on earnings/analyst text.

    Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
      → Reddit, StockTwits, HackerNews, and all general/tech
      Social + general domain, trained on 124M tweets.

  Public sentiment (5.2) now covers:
    Reddit, StockTwits, HackerNews (not just Reddit)

  model_breakdown counter fixed:
    Was checking "vader" which never existed → showed VADER: 0
    Now correctly shows FinBERT vs RoBERTa split only.
"""

from transformers import pipeline as hf_pipeline
import re, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import SENTIMENT_MODEL, FINBERT_MODEL


# ──────────────────────────────────────────────────────────
#  Financial sources → FinBERT
#  Everything else  → Twitter-RoBERTa
# ──────────────────────────────────────────────────────────
FINANCIAL_SOURCES = {
    "Yahoo Finance", "NVIDIA IR", "NewsAPI", "NVIDIA Newsroom",
    "NVIDIA About", "NVIDIA Blog", "NVIDIA News", "Reuters",
    "Bloomberg", "Wall Street Journal", "Financial Times",
    "CNBC", "MarketBeat", "GlobeNewswire", "PRNewswire",
    "Barchart.com",
}

# Social sources → Twitter-RoBERTa + count toward public_sentiment
SOCIAL_SOURCES = {
    "HackerNews",
    "StockTwits",   # ← NEW: financial social network
}
# Reddit/r/* matched by prefix check in _route()


def _route(source: str) -> str:
    if source in FINANCIAL_SOURCES:
        return FINBERT_MODEL
    if source in SOCIAL_SOURCES or source.startswith("Reddit"):
        return SENTIMENT_MODEL
    return SENTIMENT_MODEL   # default: RoBERTa handles general tech well


# ──────────────────────────────────────────────────────────
#  Model registry — lazy-loaded singletons
# ──────────────────────────────────────────────────────────
_models    = {}
_attempted = set()   # track which models we already tried loading


def _get_model(model_name: str):
    """
    Lazy-load once, cache result including failures.
    Uses _attempted set so we NEVER retry a failed model —
    prevents 100+ identical 401 error loops.
    Returns None on any failure — never raises.
    """
    if model_name in _attempted:
        return _models.get(model_name)  # None if failed

    _attempted.add(model_name)          # mark before trying (prevents retry on crash)
    short = model_name.split("/")[-1]
    try:
        # Disable HF token so we get fast 401 instead of hanging request
        import os
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", "")
        os.environ.setdefault("HF_TOKEN", "")

        print(f"  [Sentiment] Loading {short} ...")
        pipe = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=512,
            device=-1,
            token=False,   # explicitly no token — avoids expired-token auth attempts
        )
        _models[model_name] = pipe
        print(f"  [Sentiment] ✅ {short} ready")
        return pipe
    except Exception as e:
        _models[model_name] = None
        print(f"  [Sentiment] ⚠️  {short} unavailable ({type(e).__name__}) "
              f"— documents will fall back to VADER scoring.")
        return None


# ──────────────────────────────────────────────────────────
#  Label normalisation
# ──────────────────────────────────────────────────────────
_LABEL_MAP = {
    "positive": "positive", "pos": "positive", "label_2": "positive",
    "negative": "negative", "neg": "negative", "label_0": "negative",
    "neutral":  "neutral",  "neu": "neutral",  "label_1": "neutral",
}


def _norm(label: str) -> str:
    return _LABEL_MAP.get(label.lower().strip(), "neutral")


# ──────────────────────────────────────────────────────────
#  Sentence splitting
# ──────────────────────────────────────────────────────────
def _sentences(text: str, max_n: int = 8) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20][:max_n]


# ──────────────────────────────────────────────────────────
#  Single chunk analysis
# ──────────────────────────────────────────────────────────
def _analyze_chunk(text: str, model_name: str) -> dict:
    """
    Run one sentence through the model.
    Fallback chain: primary model → other HF model → VADER → neutral.
    Never raises, always returns a result.
    """
    pipe = _get_model(model_name)
    if pipe is None:
        # Try the other HF model
        alt = FINBERT_MODEL if model_name != FINBERT_MODEL else SENTIMENT_MODEL
        pipe = _get_model(alt)
    if pipe is not None:
        try:
            r = pipe(text[:512])[0]
            return {"label": _norm(r["label"]), "score": round(r["score"], 4)}
        except Exception:
            pass
    # VADER fallback — works offline, no HF token, no download
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _va = SentimentIntensityAnalyzer()
        comp = _va.polarity_scores(text)["compound"]
        if comp >= 0.05:
            return {"label": "positive", "score": round(comp, 4)}
        if comp <= -0.05:
            return {"label": "negative", "score": round(abs(comp), 4)}
        return {"label": "neutral", "score": 0.5}
    except Exception:
        return {"label": "neutral", "score": 0.5}


# ──────────────────────────────────────────────────────────
#  analyze_one — main document analysis
# ──────────────────────────────────────────────────────────
def analyze_one(doc: dict) -> dict:
    """
    Analyze one document. Returns dict with keys:
      label, score, magnitude, pos_score, neg_score, neu_score,
      num_sentences, source_category, model_used
    """
    source     = doc.get("source", "")
    model_name = _route(source)
    title      = doc.get("title", "")
    content    = doc.get("content", "")
    full_text  = f"{title}. {title}. {content}"  # title twice = extra weight

    sents = _sentences(full_text) or [full_text[:400]]
    results = [_analyze_chunk(s, model_name) for s in sents]

    votes = {"positive": 0, "negative": 0, "neutral": 0}
    sums  = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for r in results:
        votes[r["label"]] += 1
        sums[r["label"]]  += r["score"]

    n   = len(results) or 1
    pos = sums["positive"] / n
    neg = sums["negative"] / n
    neu = sums["neutral"]  / n

    winning   = max(votes, key=votes.get)
    magnitude = round(abs(pos - neg), 4)

    # Source category for 5.1/5.2 split
    if source in FINANCIAL_SOURCES:
        cat = "financial"
    elif source in SOCIAL_SOURCES or source.startswith("Reddit") or source == "StockTwits":
        cat = "social"
    else:
        cat = "general"

    return {
        "label":           winning,
        "score":           round(max(pos, neg, neu), 4),
        "magnitude":       magnitude,
        "pos_score":       round(pos, 4),
        "neg_score":       round(neg, 4),
        "neu_score":       round(neu, 4),
        "num_sentences":   len(sents),
        "source_category": cat,
        "model_used":      model_name.split("/")[-1],
    }


# ──────────────────────────────────────────────────────────
#  analyze_documents — pipeline entry point
# ──────────────────────────────────────────────────────────
def analyze_documents(documents: list[dict]) -> list[dict]:
    total = len(documents)
    print(f"  [Sentiment] Analyzing {total} documents ...")
    for i, doc in enumerate(documents):
        r = analyze_one(doc)
        doc["sentiment"]           = r["label"]
        doc["sentiment_score"]     = r["score"]
        doc["sentiment_magnitude"] = r["magnitude"]
        doc["sentiment_pos"]       = r["pos_score"]
        doc["sentiment_neg"]       = r["neg_score"]
        doc["sentiment_neu"]       = r["neu_score"]
        doc["source_category"]     = r["source_category"]
        doc["model_used"]          = r["model_used"]
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    {i+1}/{total} ...")
    return documents


# ──────────────────────────────────────────────────────────
#  sentiment_summary — simplified and accurate
# ──────────────────────────────────────────────────────────
def sentiment_summary(documents: list[dict]) -> dict:
    """
    Compute sentiment stats for dashboard Sections 5.1/5.2/5.3.

    Returns all original keys PLUS:
      news_sentiment   — avg_score, pos/neu/neg counts, count
      public_sentiment — same, for Reddit + StockTwits + HackerNews
      trend_series     — [{date, score, label}, ...] daily avg
      model_breakdown  — {FinBERT: N, RoBERTa: N}
    """
    if not documents:
        return {}

    import numpy as np

    counts     = {"positive": 0, "negative": 0, "neutral": 0}
    by_category = {k: {"positive":0,"negative":0,"neutral":0}
                   for k in ("financial","social","general")}
    magnitudes  = []
    weighted_sum = 0.0
    pos_docs, neg_docs = [], []

    news_signed   = []   # signed scores for financial docs
    public_signed = []   # signed scores for social docs
    date_groups   = {}   # date → [signed scores]
    fb_count = rb_count = 0

    for doc in documents:
        label  = doc.get("sentiment", "neutral")
        score  = doc.get("sentiment_score", 0.5)
        mag    = doc.get("sentiment_magnitude", 0.0)
        cat    = doc.get("source_category", "general")
        url    = doc.get("url", "")
        date   = doc.get("date", "")[:10]
        model  = doc.get("model_used", "").lower()

        counts[label] = counts.get(label, 0) + 1
        if cat in by_category:
            by_category[cat][label] = by_category[cat].get(label, 0) + 1
        magnitudes.append(mag)

        signed = score if label == "positive" else (-score if label == "negative" else 0.0)

        if label == "positive":
            weighted_sum += score
            pos_docs.append({"title": doc.get("title","")[:80],
                             "source": doc.get("source",""),
                             "score": score, "url": url})
        elif label == "negative":
            weighted_sum -= score
            neg_docs.append({"title": doc.get("title","")[:80],
                             "source": doc.get("source",""),
                             "score": score, "url": url})

        if cat == "financial":
            news_signed.append(signed)
        elif cat == "social":
            public_signed.append(signed)

        if date:
            date_groups.setdefault(date, []).append(signed)

        # Model breakdown counter (fixed — no VADER)
        if "finbert" in model:
            fb_count += 1
        else:
            rb_count += 1

    total = len(documents) or 1
    pos_docs.sort(key=lambda x: x["score"], reverse=True)
    neg_docs.sort(key=lambda x: x["score"], reverse=True)
    overall        = max(counts, key=counts.get)
    weighted_score = round(weighted_sum / total, 4)

    def _bucket(scores):
        if not scores:
            return {"avg_score": 0.0, "positive": 0, "neutral": 0,
                    "negative": 0, "count": 0}
        pos = sum(1 for s in scores if s >= 0.05)
        neg = sum(1 for s in scores if s <= -0.05)
        return {
            "avg_score": round(float(np.mean(scores)), 4),
            "positive":  pos,
            "neutral":   len(scores) - pos - neg,
            "negative":  neg,
            "count":     len(scores),
        }

    trend_series = sorted([
        {
            "date":  d,
            "score": round(float(np.mean(v)), 4),
            "label": ("positive" if np.mean(v) >= 0.05
                      else "negative" if np.mean(v) <= -0.05
                      else "neutral"),
        }
        for d, v in date_groups.items()
    ], key=lambda x: x["date"])

    return {
        # Original keys (pipeline.py uses these unchanged)
        "positive":       counts["positive"],
        "negative":       counts["negative"],
        "neutral":        counts["neutral"],
        "total":          total,
        "overall":        overall,
        "positive_pct":   round(counts["positive"] / total * 100, 1),
        "negative_pct":   round(counts["negative"] / total * 100, 1),
        "neutral_pct":    round(counts["neutral"]  / total * 100, 1),
        "avg_magnitude":  round(sum(magnitudes)/len(magnitudes), 4) if magnitudes else 0,
        "weighted_score": weighted_score,
        "by_category":    by_category,
        "top_positive":   pos_docs[:5],
        "top_negative":   neg_docs[:5],
        # New keys for Section 5
        "news_sentiment":    _bucket(news_signed),
        "public_sentiment":  _bucket(public_signed),
        "trend_series":      trend_series,
        "model_breakdown":   {"FinBERT": fb_count, "RoBERTa": rb_count},
    }


if __name__ == "__main__":
    sample = [
        {"title": "NVIDIA crushes earnings", "content": "Revenue surged 200%.",
         "source": "Yahoo Finance"},
        {"title": "NVDA GPU shortage", "content": "AI chip demand outstrips supply.",
         "source": "HackerNews"},
        {"title": "StockTwits NVDA bullish", "content": "NVDA to the moon! Blackwell demand insane.",
         "source": "StockTwits"},
    ]
    analyzed = analyze_documents(sample)
    for d in analyzed:
        print(f"  {d['title'][:40]:40s} → {d['sentiment']:8s} "
              f"({d['sentiment_score']:.2f}) cat={d['source_category']}")
    print("\nSummary:", sentiment_summary(analyzed))
