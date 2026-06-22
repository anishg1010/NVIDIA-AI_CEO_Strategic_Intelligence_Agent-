"""
============================================================
  config/settings.py — Master Configuration
  NVIDIA AI CEO Strategic Intelligence Agent
============================================================
  Every tuneable parameter lives here with technical reasoning.
============================================================
"""

import os
from dotenv import load_dotenv # loading api key from local. not pushing
load_dotenv() 
# ════════════════════════════════════════════════════════
#  LLM — Ollama local
# ════════════════════════════════════════════════════════
#
#  LLM_MODEL:
#    "llama3.1:8b"  — best overall quality/speed balance at 8B.
#                     Strong instruction following, good JSON output.
#    "mistral:7b"   — slightly faster, slightly weaker reasoning.
#    "qwen2.5:7b"   — good alternative, especially for structured output.
#
#  LLM_TEMPERATURE = 0.1:
#    Analysis tasks (opportunity/risk/trend extraction) need
#    consistency over creativity. 0.1 gives near-deterministic
#    structured JSON output. Don't go above 0.3 for JSON tasks
#    or the model starts inventing fields.
#
#  LLM_TOP_P = 0.9:
#    Standard nucleus sampling. Keeps the probability mass
#    concentrated on likely tokens. Leave at 0.9.
#
#  LLM_MAX_TOKENS = 2048:
#    The intelligence engine asks for 5 opportunities/risks/trends
#    each as a JSON object. Each object is ~200 tokens → 5 × 200 = 1000.
#    1024 sometimes truncates the closing "]". 2048 is safe.
#
#  OLLAMA_BASE_URL:
#    Default Ollama port. Only change if you moved it.

LLM_MODEL        = "llama3.1:8b"
LLM_TEMPERATURE  = 0.1
LLM_TOP_P        = 0.9
LLM_MAX_TOKENS   = 2048          # was 1024 — raised to prevent JSON truncation
OLLAMA_BASE_URL  = "http://localhost:11434"


# ════════════════════════════════════════════════════════
#  Embedding Model
# ════════════════════════════════════════════════════════
#
#  EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5":
#    768-dim, strong semantic retrieval, good for business text.
#    Outperforms MiniLM on domain-specific retrieval benchmarks
#    (MTEB: 63.6 vs 56.3 for MiniLM-L6).
#    Tradeoff: slightly slower than MiniLM (~2x), still runs fine on CPU.
#
#  Alternative: "all-MiniLM-L6-v2"
#    384-dim, 2x faster, slightly lower quality.
#    Swap if pipeline is too slow on your machine.
#
#  EMBEDDING_DIM must match the model:
#    bge-base-en-v1.5  → 768
#    all-MiniLM-L6-v2  → 384

EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM    = 768


# ════════════════════════════════════════════════════════
#  ChromaDB Vector Store
# ════════════════════════════════════════════════════════
#
#  TOP_K_RETRIEVAL = 8:
#    How many chunks to pull per RAG query.
#    8 chunks × ~400 chars = ~3200 chars of context.
#    At ~4 chars/token → ~800 tokens of context.
#    llama3.1:8b has 8192 token context window, so 8 is safe
#    and gives the LLM enough signal without filling the window.
#    Raising to 12-15 gives richer context but slower inference.

CHROMA_PERSIST_DIR = "./data/chroma_db"
CHROMA_COLLECTION  = "nvidia_intelligence"
TOP_K_RETRIEVAL    = 8         


# ════════════════════════════════════════════════════════
#  Sentiment Models
# ════════════════════════════════════════════════════════
#
#  Two-model routing based on document source:
#
#  FINBERT_MODEL = "yiyanghkust/finbert-tone":
#    Fine-tuned on financial communications (earnings calls,
#    analyst reports, press releases). 3-class output:
#    positive/neutral/negative. Better than ProsusAI/finbert
#    on tone classification (acc: 88% vs 85% on FPB benchmark).
#    Used for: Yahoo Finance, NVIDIA IR, NewsAPI, Bloomberg etc.
#
#  SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest":
#    RoBERTa fine-tuned on 124M tweets. Best for short informal text,
#    community posts, tech forum discussions.
#    Used for: Reddit, HackerNews, general web.
#
#  Both models output positive/neutral/negative labels.

FINBERT_MODEL   = "yiyanghkust/finbert-tone"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# ════════════════════════════════════════════════════════
#  Data Collection
# ════════════════════════════════════════════════════════
#
#  MAX_DOCUMENTS = 300:
#    FIX: was set to 10 — only collecting Yahoo Finance RSS.
#    300 gives a good corpus: ~200 news + ~80 Reddit + ~20 other.
#    With CHUNK_SIZE=400, 300 docs → ~1500-2000 chunks in ChromaDB.
#    Pipeline takes ~3-5 min on first run.
#
#  CHUNK_SIZE = 400:
#    FIX: was set to 100 chars — too small.
#    At 100 chars, each chunk is ~25 words. The LLM gets no coherent
#    context. 400 chars (~100 words) is the standard RAG chunk size
#    that preserves sentence-level meaning.
#
#  CHUNK_OVERLAP = 80:
#    20% overlap prevents information loss at chunk boundaries.
#    Standard practice: 15-25% of chunk size.

COMPANY_NAME      = "NVIDIA"
COMPANY_TICKER    = "NVDA"
MAX_DOCUMENTS     = 300      # FIX: was 10

RSS_FEEDS = [
    "https://nvidianews.nvidia.com/rss/all",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    "https://feeds.feedburner.com/TechCrunch",
    "https://www.theverge.com/rss/index.xml",
    "https://www.electronicdesign.com/rss.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.wired.com/feed/rss",
    "https://www.tomshardware.com/feeds/all",
]

REDDIT_SUBREDDITS = [
    "nvidia",
    "artificial",
    "MachineLearning",
    "stocks",
    "hardware",
    "Futurology",
]
REDDIT_POST_LIMIT = 25       # per subreddit; 25 × 6 = 150 posts max

NEWS_API_KEY      = os.environ.get("NEWS_API_KEY", "d18749108ef94458aa73b8f16e10f6cc")
NEWS_QUERY        = "NVIDIA GPU AI semiconductor"
NEWS_MAX_RESULTS  = 50

HACKERNEWS_QUERY  = "NVIDIA"
HACKERNEWS_LIMIT  = 30

CHUNK_SIZE        = 400      # FIX: was 100 — too small for coherent RAG context
CHUNK_OVERLAP     = 80


# ════════════════════════════════════════════════════════
#  Intelligence Engine
# ════════════════════════════════════════════════════════
#
#  NUM_OPPORTUNITIES/RISKS/TRENDS = 5:
#    FIX: was set to 1 — barely any intelligence output.
#    5 items per category gives the CEO meaningful signal.
#    The LLM prompt asks for exactly N items; setting to 1
#    means a single risk/opportunity regardless of what's in the data.
#
#  OPPORTUNITY/RISK_THRESHOLD = 0.65:
#    Minimum confidence_score for an item to be included.
#    0.65 filters out weak signals while keeping meaningful ones.

OPPORTUNITY_THRESHOLD = 0.65
RISK_THRESHOLD        = 0.65
NUM_OPPORTUNITIES     = 5    # FIX: was 1
NUM_RISKS             = 5    # FIX: was 1
NUM_TRENDS            = 5    # FIX: was 1


# ════════════════════════════════════════════════════════
#  Dashboard
# ════════════════════════════════════════════════════════

DASHBOARD_TITLE      = "NVIDIA — CEO Strategic Intelligence Dashboard"
AUTO_REFRESH_SECONDS = 300
