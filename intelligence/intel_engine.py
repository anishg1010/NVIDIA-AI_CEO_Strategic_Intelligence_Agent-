"""
============================================================
  MODULE 4 — intelligence/intel_engine.py
  Responsibility: Identify Opportunities, Risks & Trends
                  using RAG (ChromaDB → Ollama LLM)
============================================================

  For each category (opportunity / risk / trend), we:
    1. Query ChromaDB with a targeted search prompt
    2. Build a context string from the top-k chunks
    3. Ask the Ollama LLM to extract structured insights
    4. Parse and return structured results
"""

import json
import ollama

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
    NUM_OPPORTUNITIES, NUM_RISKS, NUM_TRENDS,
    OPPORTUNITY_THRESHOLD, RISK_THRESHOLD, TOP_K_RETRIEVAL,
    COMPANY_NAME
)
from processor.knowledge_store import query as kb_query


# ──────────────────────────────────────────
#  Core RAG + LLM call
# ──────────────────────────────────────────
def _rag_ask(search_query: str, system_prompt: str, user_prompt: str) -> str:
    """
    1. Retrieve relevant chunks from ChromaDB.
    2. Build context.
    3. Call Ollama LLM with system + user prompt.

    Returns the raw LLM text response.
    """
    # Step 1: Retrieve context
    chunks = kb_query(search_query, top_k=TOP_K_RETRIEVAL)

    # Step 2: Format context (include source attribution)
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {c['source']} | {c['date']}]\n{c['text']}"
        )
    context = "\n\n".join(context_parts)

    # Step 3: Call LLM via Ollama
    full_user_prompt = f"""
CONTEXT FROM KNOWLEDGE BASE:
{context}

{user_prompt}
"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            options={
                "temperature": LLM_TEMPERATURE,   # tune in settings.py
                "top_p":       LLM_TOP_P,
                "num_predict": LLM_MAX_TOKENS,
            },
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": full_user_prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[LLM Error: {e}]"


# ──────────────────────────────────────────
#  Parse JSON from LLM response
# ──────────────────────────────────────────
def _parse_json_list(text: str, fallback_key: str = "items") -> list[dict]:
    """
    Extract a JSON list from LLM output (handles markdown code blocks).
    Returns an empty list on failure instead of crashing.
    """
    try:
        # Strip markdown code fences if present
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Try to find a JSON array
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return []


# ──────────────────────────────────────────
#  Opportunity Detection
# ──────────────────────────────────────────
def identify_opportunities(n: int = NUM_OPPORTUNITIES) -> list[dict]:
    """
    Find strategic opportunities for NVIDIA from the knowledge base.

    Hyperparameters to tune:
      NUM_OPPORTUNITIES  → how many to surface
      LLM_TEMPERATURE    → lower = more focused analysis
    """
    print("  [Intel] Identifying opportunities...")

    system = f"""You are a strategic business analyst for {COMPANY_NAME}.
Your job is to identify real business opportunities from news and market data.
Always respond with valid JSON only — no explanation text before or after."""

    user = f"""Based on the context above, identify the top {n} strategic opportunities for {COMPANY_NAME}.

Return a JSON array with exactly this structure:
[
  {{
    "title": "Short opportunity title",
    "description": "2-3 sentence explanation of the opportunity",
    "impact_level": "High" | "Medium" | "Low",
    "confidence_score": 0.0 to 1.0,
    "evidence": ["quote or fact from source 1", "quote or fact from source 2"],
    "category": "Technology" | "Market" | "Partnership" | "Product"
  }}
]"""

    raw = _rag_ask(
        search_query=f"{COMPANY_NAME} opportunities growth AI GPU market expansion",
        system_prompt=system,
        user_prompt=user,
    )
    items = _parse_json_list(raw)

    # Add type tag for dashboard filtering
    for item in items:
        item["type"] = "opportunity"

    return items[:n]


# ──────────────────────────────────────────
#  Risk Detection
# ──────────────────────────────────────────
def identify_risks(n: int = NUM_RISKS) -> list[dict]:
    """
    Find strategic risks for NVIDIA from the knowledge base.

    Hyperparameters to tune:
      NUM_RISKS        → how many to surface
      RISK_THRESHOLD   → min similarity score (set in settings.py)
    """
    print("  [Intel] Identifying risks...")

    system = f"""You are a risk analyst for {COMPANY_NAME}.
Your job is to identify real threats and risks from news and market data.
Always respond with valid JSON only."""

    user = f"""Based on the context above, identify the top {n} strategic risks for {COMPANY_NAME}.

Return a JSON array with exactly this structure:
[
  {{
    "title": "Short risk title",
    "description": "2-3 sentence explanation of the risk",
    "severity": "Critical" | "High" | "Medium" | "Low",
    "confidence_score": 0.0 to 1.0,
    "evidence": ["fact from source 1", "fact from source 2"],
    "category": "Competitive" | "Regulatory" | "Market" | "Operational" | "Geopolitical"
  }}
]"""

    raw = _rag_ask(
        search_query=f"{COMPANY_NAME} risks competition regulation export control AMD Intel",
        system_prompt=system,
        user_prompt=user,
    )
    items = _parse_json_list(raw)

    for item in items:
        item["type"] = "risk"

    return items[:n]


# ──────────────────────────────────────────
#  Trend Detection
# ──────────────────────────────────────────
def identify_trends(n: int = NUM_TRENDS) -> list[dict]:
    """
    Detect emerging trends relevant to NVIDIA.

    Hyperparameter to tune:
      NUM_TRENDS → how many trends to surface
    """
    print("  [Intel] Identifying trends...")

    system = f"""You are a technology trend analyst covering {COMPANY_NAME}'s industry.
Your job is to identify emerging trends from news data.
Always respond with valid JSON only."""

    user = f"""Based on the context above, identify {n} emerging trends most relevant to {COMPANY_NAME}.

Return a JSON array:
[
  {{
    "title": "Trend name",
    "description": "2-3 sentence summary of the trend",
    "relevance": "High" | "Medium" | "Low",
    "time_horizon": "Short-term (0-6 months)" | "Medium-term (6-18 months)" | "Long-term (18+ months)",
    "evidence": ["supporting data point 1", "supporting data point 2"],
    "category": "Technology" | "Market" | "Customer" | "Regulatory"
  }}
]"""

    raw = _rag_ask(
        search_query="AI trends generative AI data center chips semiconductor technology 2024 2025",
        system_prompt=system,
        user_prompt=user,
    )
    items = _parse_json_list(raw)

    for item in items:
        item["type"] = "trend"

    return items[:n]


# ──────────────────────────────────────────
#  Run all three at once
# ──────────────────────────────────────────
def run_intelligence_engine() -> dict:
    """
    Run all analyses and return a combined intelligence report.

    Returns:
      {
        "opportunities": [...],
        "risks":         [...],
        "trends":        [...],
      }
    """
    return {
        "opportunities": identify_opportunities(),
        "risks":         identify_risks(),
        "trends":        identify_trends(),
    }


if __name__ == "__main__":
    report = run_intelligence_engine()
    print("\nOpportunities:", len(report["opportunities"]))
    print("Risks:",         len(report["risks"]))
    print("Trends:",        len(report["trends"]))
    print("\nFirst opportunity:", report["opportunities"][0] if report["opportunities"] else "None")
