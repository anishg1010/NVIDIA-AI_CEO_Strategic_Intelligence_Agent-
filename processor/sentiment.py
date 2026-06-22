"""
============================================================
  MODULE 3 — processor/sentiment.py  (IMPROVED)
============================================================

  Improvements over original:
    1. Dual-model approach:
       - FinBERT for financial/business articles (IR, Yahoo Finance)
       - RoBERTa-twitter for social content (Reddit, HackerNews)
       - Auto-selects based on document source
    2. Sentence-level sentiment (not just first 512 chars)
       - Splits long docs into sentences, averages scores
    3. Richer output per document:
       - sentiment, sentiment_score, sentiment_magnitude,
         source_category, num_sentences_analyzed
    4. Richer summary:
       - breakdown by source category (financial vs social)
       - top positive and negative articles
       - sentiment trend (if date info available)
       - magnitude-weighted overall score
"""

from transformers import pipeline as hf_pipeline
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import SENTIMENT_MODEL, FINBERT_MODEL


# ──────────────────────────────────────────
#  Model registry — lazy loaded singletons
# ──────────────────────────────────────────
_models = {}

# Source categories for model routing
FINANCIAL_SOURCES = {
    "Yahoo Finance", "NVIDIA IR", "NewsAPI", "NVIDIA Newsroom",
    "NVIDIA About", "NVIDIA Blog", "Reuters", "Bloomberg",
    "Wall Street Journal", "Financial Times", "CNBC"
}

SOCIAL_SOURCES = {
    "HackerNews", "Wikipedia"  # neutral/social
}
# Reddit and RSS use general model


def _get_model(model_name: str):
    """Lazy-load a model once and cache it."""
    if model_name not in _models:
        print(f"  [Sentiment] Loading model: {model_name}")
        _models[model_name] = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=512,
            device=-1,  # CPU; set to 0 for GPU
        )
    return _models[model_name]


def _route_model(source: str) -> str:
    """
    Choose the best sentiment model based on document source.
    
    Hyperparameters to tune (in settings.py):
      SENTIMENT_MODEL  → general/social model
      FINBERT_MODEL    → financial text model
    """
    if source in FINANCIAL_SOURCES:
        return FINBERT_MODEL    # ProsusAI/finbert — tuned for financial text
    return SENTIMENT_MODEL       # RoBERTa — general purpose


# ──────────────────────────────────────────
#  Normalize labels across models
# ──────────────────────────────────────────
def _normalize_label(label: str) -> str:
    """Different models use different label names — standardize them."""
    label = label.lower().strip()
    
    # FinBERT uses: positive, negative, neutral
    # RoBERTa uses: LABEL_0 (neg), LABEL_1 (neu), LABEL_2 (pos)
    # Twitter-RoBERTa: negative, neutral, positive
    mapping = {
        "positive": "positive", "pos": "positive", "label_2": "positive",
        "negative": "negative", "neg": "negative", "label_0": "negative",
        "neutral":  "neutral",  "neu": "neutral",  "label_1": "neutral",
    }
    return mapping.get(label, "neutral")


# ──────────────────────────────────────────
#  Split into sentences for better coverage
# ──────────────────────────────────────────
def _split_sentences(text: str, max_sentences: int = 8) -> list[str]:
    """
    Split text into sentences. Analyze up to max_sentences
    to avoid exceeding model context limits.
    
    Hyperparameter: max_sentences (default 8)
    Tune: raise for more thorough analysis, lower for speed.
    """
    # Simple sentence splitter on . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences[:max_sentences]


# ──────────────────────────────────────────
#  Analyze a single text chunk
# ──────────────────────────────────────────
def _analyze_chunk(text: str, model_name: str) -> dict:
    """Run one text through a model, return normalized result."""
    try:
        pipe   = _get_model(model_name)
        result = pipe(text[:512])[0]
        return {
            "label": _normalize_label(result["label"]),
            "score": round(result["score"], 4),
        }
    except Exception:
        return {"label": "neutral", "score": 0.5}


# ──────────────────────────────────────────
#  Main: analyze one document
# ──────────────────────────────────────────
def analyze_one(doc: dict) -> dict:
    """
    Analyze a document using sentence-level averaging.
    
    Returns enriched sentiment dict:
      {
        label:              positive | neutral | negative
        score:              confidence of winning label (0-1)
        magnitude:          how strong the sentiment is (0-1)
        pos_score:          raw positive probability
        neg_score:          raw negative probability
        neu_score:          raw neutral probability
        num_sentences:      how many sentences were analyzed
        source_category:    financial | social | general
        model_used:         which model
      }
    """
    source     = doc.get("source", "")
    model_name = _route_model(source)

    # Combine title + content, with title weighted (repeated)
    title   = doc.get("title", "")
    content = doc.get("content", "")
    full_text = f"{title}. {title}. {content}"  # title twice = more weight

    sentences = _split_sentences(full_text)
    if not sentences:
        sentences = [full_text[:400]]

    # Analyze each sentence
    results = [_analyze_chunk(s, model_name) for s in sentences]

    # Aggregate: count votes and average scores
    vote_counts = {"positive": 0, "negative": 0, "neutral": 0}
    sum_scores  = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    for r in results:
        lbl = r["label"]
        vote_counts[lbl] += 1
        sum_scores[lbl]  += r["score"]

    n = len(results) or 1

    # Normalize to probabilities
    pos = sum_scores["positive"] / n
    neg = sum_scores["negative"] / n
    neu = sum_scores["neutral"]  / n

    # Winning label = majority vote (not just score average)
    winning = max(vote_counts, key=vote_counts.get)

    # Magnitude = how far from neutral (0=neutral, 1=strongly polar)
    magnitude = round(abs(pos - neg), 4)

    # Source category
    if source in FINANCIAL_SOURCES:
        src_cat = "financial"
    elif any(r in source for r in ["Reddit", "HackerNews"]):
        src_cat = "social"
    else:
        src_cat = "general"

    return {
        "label":            winning,
        "score":            round(max(pos, neg, neu), 4),
        "magnitude":        magnitude,
        "pos_score":        round(pos, 4),
        "neg_score":        round(neg, 4),
        "neu_score":        round(neu, 4),
        "num_sentences":    len(sentences),
        "source_category":  src_cat,
        "model_used":       model_name.split("/")[-1],
    }


# ──────────────────────────────────────────
#  Analyze a list of documents
# ──────────────────────────────────────────
def analyze_documents(documents: list[dict]) -> list[dict]:
    """
    Add sentiment fields to each document dict in-place.
    Prints progress every 10 docs.
    """
    total = len(documents)
    print(f"  [Sentiment] Analyzing {total} documents (dual-model)...")

    for i, doc in enumerate(documents):
        result = analyze_one(doc)
        doc["sentiment"]           = result["label"]
        doc["sentiment_score"]     = result["score"]
        doc["sentiment_magnitude"] = result["magnitude"]
        doc["sentiment_pos"]       = result["pos_score"]
        doc["sentiment_neg"]       = result["neg_score"]
        doc["sentiment_neu"]       = result["neu_score"]
        doc["source_category"]     = result["source_category"]
        doc["model_used"]          = result["model_used"]

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    {i+1}/{total} analyzed...")

    return documents


# ──────────────────────────────────────────
#  Rich sentiment summary
# ──────────────────────────────────────────
def sentiment_summary(documents: list[dict]) -> dict:
    """
    Compute detailed sentiment stats for the dashboard.

    Returns a rich dict including:
      - overall counts and percentages
      - breakdown by source category (financial vs social vs general)
      - top 5 most positive articles
      - top 5 most negative articles
      - average magnitude (how strongly-felt the sentiment is)
      - weighted overall score (-1 to +1)
    """
    if not documents:
        return {}

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    by_category = {
        "financial": {"positive": 0, "negative": 0, "neutral": 0},
        "social":    {"positive": 0, "negative": 0, "neutral": 0},
        "general":   {"positive": 0, "negative": 0, "neutral": 0},
    }

    magnitudes   = []
    weighted_sum = 0.0
    pos_docs, neg_docs = [], []

    # ── New accumulators ─────────────────────────────
    news_scores   = []
    public_scores = []
    date_groups   = {}   # date_str → [scores]

    for doc in documents:
        label    = doc.get("sentiment", "neutral")
        score    = doc.get("sentiment_score", 0.5)
        mag      = doc.get("sentiment_magnitude", 0.0)
        cat      = doc.get("source_category", "general")
        url      = doc.get("url", "")
        date_str = doc.get("date", "")[:10]

        counts[label] = counts.get(label, 0) + 1

        if cat in by_category:
            by_category[cat][label] = by_category[cat].get(label, 0) + 1

        magnitudes.append(mag)

        # Signed score for averages: positive=+score, negative=-score
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

        # News vs public split
        if cat == "financial":
            news_scores.append(signed)
        elif cat == "social":
            public_scores.append(signed)

        # Trend series aggregation by date
        if date_str:
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append(signed)

    total = len(documents) or 1

    pos_docs.sort(key=lambda x: x["score"], reverse=True)
    neg_docs.sort(key=lambda x: x["score"], reverse=True)
    overall        = max(counts, key=counts.get)
    weighted_score = round(weighted_sum / total, 4)

    # ── news_sentiment / public_sentiment buckets ────
    def _bucket(scores):
        if not scores:
            return {"avg_score": 0.0, "positive": 0, "neutral": 0, "negative": 0, "count": 0}
        import numpy as np
        pos = sum(1 for s in scores if s >= 0.05)
        neg = sum(1 for s in scores if s <= -0.05)
        return {
            "avg_score": round(float(np.mean(scores)), 4),
            "positive":  pos,
            "neutral":   len(scores) - pos - neg,
            "negative":  neg,
            "count":     len(scores),
        }

    # ── trend_series: [{date, score, label}] sorted ──
    trend_series = sorted([
        {
            "date":  d,
            "score": round(sum(v)/len(v), 4),
            "label": ("positive" if sum(v)/len(v) >= 0.05
                      else "negative" if sum(v)/len(v) <= -0.05
                      else "neutral"),
        }
        for d, v in date_groups.items()
    ], key=lambda x: x["date"])

    # ── model_breakdown ──────────────────────────────
    model_usage = {"FinBERT": 0, "RoBERTa": 0, "VADER": 0}
    for doc in documents:
        m = doc.get("model_used", "")
        if "finbert" in m.lower() or "finbert" in m:
            model_usage["FinBERT"] += 1
        elif "roberta" in m.lower():
            model_usage["RoBERTa"] += 1
        else:
            model_usage["VADER"] += 1

    return {
        # ── Original keys — all preserved ──
        "positive":       counts["positive"],
        "negative":       counts["negative"],
        "neutral":        counts["neutral"],
        "total":          total,
        "overall":        overall,
        "positive_pct":   round(counts["positive"] / total * 100, 1),
        "negative_pct":   round(counts["negative"] / total * 100, 1),
        "neutral_pct":    round(counts["neutral"]  / total * 100, 1),
        "avg_magnitude":  round(sum(magnitudes) / len(magnitudes), 4) if magnitudes else 0,
        "weighted_score": weighted_score,
        "by_category":    by_category,
        "top_positive":   pos_docs[:5],
        "top_negative":   neg_docs[:5],
        # ── New keys for Section 5 ──
        "news_sentiment":   _bucket(news_scores),
        "public_sentiment": _bucket(public_scores),
        "trend_series":     trend_series,
        "model_breakdown":  model_usage,
    }


if __name__ == "__main__":
    sample = [
        {"title": "NVIDIA crushes earnings expectations",
         "content": "Revenue surged 200% driven by AI GPU demand. Analysts raise targets.",
         "source": "Yahoo Finance"},
        {"title": "NVIDIA faces China export restrictions",
         "content": "US government restricts sale of H100 chips to China. Major revenue risk.",
         "source": "NVIDIA IR"},
        {"title": "CUDA 12 released with new features",
         "content": "Developers excited about new parallel computing improvements.",
         "source": "HackerNews"},
    ]
    analyzed = analyze_documents(sample)
    for d in analyzed:
        print(f"  {d['title'][:40]:40s} → {d['sentiment']:8s} ({d['sentiment_score']:.2f}) mag={d['sentiment_magnitude']:.2f}")
    print("\nSummary:", sentiment_summary(analyzed))
