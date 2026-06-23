"""
============================================================
  pipeline.py — Master Orchestrator  (FIXED)
============================================================

  BUG FIX: total_documents was showing 964 (same as chunks)
  because when collection was skipped, we used:
    "total_documents": db_stats["total_chunks"]   ← WRONG
  
  Fix: We now store doc_count separately in report.json and
  reload it on subsequent runs. Chunks and documents are
  now always distinct numbers.

  Run:
    python pipeline.py          ← uses cached DB if exists
    python pipeline.py --force  ← re-collects everything
"""

import json
import os
from datetime import datetime

from collector.data_collector  import collect_all
from processor.knowledge_store import store_documents, get_stats
from processor.sentiment       import analyze_documents, sentiment_summary
from intelligence.intel_engine import run_intelligence_engine
from agent.ceo_agent           import generate_recommendations, generate_ceo_briefing
from config.settings           import COMPANY_NAME

REPORT_PATH = "./data/report.json"


def run_pipeline(force_recollect: bool = False) -> dict:

    os.makedirs("./data", exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  NVIDIA AI CEO Agent — Pipeline Starting")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    db_stats = get_stats()

    # ── STEP 1: Collection & Storage ────────────────────
    if db_stats["total_chunks"] > 0 and not force_recollect:
        print(f"  [Pipeline] ChromaDB has {db_stats['total_chunks']} chunks.")
        print("  [Pipeline] Skipping collection. Run with --force to re-collect.\n")
        documents      = []
        fresh_run      = False
    else:
        print("  [Pipeline] Step 1: Collecting data from 8 sources...")
        documents = collect_all()

        print("\n  [Pipeline] Step 2: Sentiment analysis...")
        documents = analyze_documents(documents)

        print("\n  [Pipeline] Step 3: Storing in ChromaDB...")
        store_documents(documents)
        fresh_run = True

    # ── STEP 2: Build metadata for dashboard ────────────
    if fresh_run and documents:
        sent_summary   = sentiment_summary(documents)

        # ── Filter trend_series to last 30 days only ──────
        # Prevents 2022-2023 stale articles from inflating volatility
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if "trend_series" in sent_summary:
            sent_summary["trend_series"] = [
                p for p in sent_summary["trend_series"]
                if p.get("date", "") >= cutoff
            ]

        documents_meta = [
            {
                "title":           d["title"],
                "source":          d["source"],
                "date":            d["date"],
                "url":             d.get("url",""),
                "sentiment":       d.get("sentiment", "neutral"),
                "sentiment_score": d.get("sentiment_score", 0.5),
            }
            for d in documents
        ]
        total_documents = len(documents)
        sources_list    = list({d["source"] for d in documents})
    else:
        # Reload from previous report — keeps document count correct
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH) as f:
                old = json.load(f)
            sent_summary    = old.get("sentiment_summary", {})
            documents_meta  = old.get("documents_meta", [])
            total_documents = old.get("meta", {}).get("total_documents", 0)
            sources_list    = old.get("meta", {}).get("sources", [])
        else:
            sent_summary    = {}
            documents_meta  = []
            total_documents = 0
            sources_list    = []

    # ── STEP 3: Intelligence Engine ─────────────────────
    print("\n  [Pipeline] Step 4: Running intelligence engine...")
    intelligence = run_intelligence_engine()

    # ── STEP 4: CEO Agent ───────────────────────────────
    print("\n  [Pipeline] Step 5: Generating CEO recommendations...")
    recommendations = generate_recommendations(intelligence)

    print("\n  [Pipeline] Step 6: Writing CEO briefing...")
    briefing = generate_ceo_briefing(intelligence, recommendations, sent_summary)

    # ── STEP 5: Assemble report ─────────────────────────
    db_stats = get_stats()   # refresh after any new inserts

    report = {
        "meta": {
            "company":          COMPANY_NAME,
            "generated_at":     datetime.now().isoformat(),
            # FIX: total_documents = real doc count, NOT chunk count
            "total_documents":  total_documents,
            "total_chunks":     db_stats["total_chunks"],
            "sources":          sources_list,
            "num_sources":      len(sources_list),
        },
        "sentiment_summary":  sent_summary,
        "documents_meta":     documents_meta,
        "opportunities":      intelligence["opportunities"],
        "risks":              intelligence["risks"],
        "trends":             intelligence["trends"],
        "recommendations":    recommendations,
        "ceo_briefing":       briefing,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Documents collected : {total_documents}")
    print(f"  Chunks in ChromaDB  : {db_stats['total_chunks']}")
    print(f"  Sources             : {len(sources_list)}")
    print(f"  Opportunities       : {len(intelligence['opportunities'])}")
    print(f"  Risks               : {len(intelligence['risks'])}")
    print(f"  Recommendations     : {len(recommendations)}")
    print(f"  Report saved to     : {REPORT_PATH}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NVIDIA AI CEO Agent Pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Force re-collection of all data")
    args = parser.parse_args()

    report = run_pipeline(force_recollect=args.force)
    print("\nCEO BRIEFING PREVIEW:")
    print("-" * 40)
    print(report["ceo_briefing"][:600] + "...")
