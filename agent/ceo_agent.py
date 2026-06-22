"""
============================================================
  MODULE 5 — agent/ceo_agent.py
  Responsibility: The AI CEO Agent
                  → Analyze intelligence
                  → Generate strategic recommendations
                  → Write executive CEO briefing
============================================================

  This is the "brain" of the system.
  It takes the outputs of intel_engine.py and uses the LLM
  to reason at an executive level: "What should we do next?"
"""

import json
import ollama

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
    COMPANY_NAME, TOP_K_RETRIEVAL
)
from processor.knowledge_store import query as kb_query


# ──────────────────────────────────────────
#  LLM call wrapper
# ──────────────────────────────────────────
def _llm(system: str, user: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    """
    Call Ollama with the configured model.

    Hyperparameters to tune:
      LLM_MODEL       → model name (e.g. "llama3.1:8b", "qwen3:8b")
      LLM_TEMPERATURE → lower = more focused, higher = more creative
      LLM_TOP_P       → nucleus sampling
      LLM_MAX_TOKENS  → max response length
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            options={
                "temperature": LLM_TEMPERATURE,
                "top_p":       LLM_TOP_P,
                "num_predict": max_tokens,
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[LLM Error: {e}]"


# ──────────────────────────────────────────
#  Parse JSON safely
# ──────────────────────────────────────────
def _parse_json(text: str) -> list | dict:
    try:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return []


# ──────────────────────────────────────────
#  Generate Strategic Recommendations
# ──────────────────────────────────────────
def generate_recommendations(intelligence: dict) -> list[dict]:
    """
    Takes the full intelligence report and generates
    prioritized, evidence-backed CEO recommendations.

    Args:
      intelligence: output of intel_engine.run_intelligence_engine()

    Returns:
      list of recommendation dicts
    """
    print("  [CEO Agent] Generating strategic recommendations...")

    # Build a compact intelligence summary for the LLM
    opp_text  = "\n".join([f"- {o['title']}: {o['description']}" for o in intelligence.get("opportunities", [])])
    risk_text = "\n".join([f"- {r['title']}: {r['description']}" for r in intelligence.get("risks", [])])
    trend_text= "\n".join([f"- {t['title']}: {t['description']}" for t in intelligence.get("trends", [])])

    # Also pull fresh RAG context
    recent_chunks = kb_query(f"{COMPANY_NAME} strategy earnings revenue competition", top_k=6)
    rag_context   = "\n\n".join([f"[{c['source']}] {c['text']}" for c in recent_chunks])

    system = f"""You are the Chief Strategy Officer advising the CEO of {COMPANY_NAME}.
Your role is to translate intelligence into concrete, prioritized strategic actions.
Think like a board-level executive, not a researcher.
Always respond with valid JSON only — no other text."""

    user = f"""
OPPORTUNITIES IDENTIFIED:
{opp_text}

RISKS IDENTIFIED:
{risk_text}

TRENDS TO WATCH:
{trend_text}

RECENT MARKET DATA:
{rag_context}

Based on the above, generate 5 strategic recommendations for {COMPANY_NAME}'s CEO.

Return a JSON array with exactly this structure:
[
  {{
    "recommendation": "Clear, actionable strategic recommendation (1-2 sentences)",
    "priority": "High" | "Medium" | "Low",
    "rationale": "Why this action makes sense now (2-3 sentences)",
    "supporting_evidence": [
      "Specific fact or data point 1",
      "Specific fact or data point 2",
      "Specific fact or data point 3"
    ],
    "expected_impact": {{
      "revenue": "Expected revenue impact",
      "market":  "Expected market position impact",
      "timeline": "Expected time to see results"
    }},
    "risk_assessment": {{
      "financial": "Financial risk level and description",
      "operational": "Operational risk level and description",
      "strategic": "Strategic risk level and description"
    }},
    "risk_level": "High" | "Medium" | "Low"
  }}
]"""

    raw = _llm(system, user, max_tokens=2000)
    recommendations = _parse_json(raw)

    # Sort by priority
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "Low"), 3))

    return recommendations


# ──────────────────────────────────────────
#  Generate CEO Briefing
# ──────────────────────────────────────────
def generate_ceo_briefing(
    intelligence: dict,
    recommendations: list[dict],
    sentiment_summary: dict,
) -> str:
    """
    Generate a concise executive briefing answering:
      1. What happened?
      2. Why does it matter?
      3. What should management do next?

    Args:
      intelligence:      output of intel_engine
      recommendations:   output of generate_recommendations
      sentiment_summary: output of sentiment.sentiment_summary

    Returns:
      Formatted briefing as a plain string
    """
    print("  [CEO Agent] Writing CEO briefing...")

    # Compact summaries
    top_opp  = intelligence["opportunities"][0]["title"] if intelligence.get("opportunities") else "N/A"
    top_risk = intelligence["risks"][0]["title"]         if intelligence.get("risks")         else "N/A"
    top_rec  = recommendations[0]["recommendation"]      if recommendations                   else "N/A"
    sentiment= sentiment_summary.get("overall", "neutral")
    pos_pct  = sentiment_summary.get("positive_pct", 0)
    neg_pct  = sentiment_summary.get("negative_pct", 0)

    system = f"""You are writing a one-page briefing for the CEO of {COMPANY_NAME}.
Be direct, decisive, and executive in tone. Use bullet points where appropriate.
Write in plain text — no markdown headers, no hashtags."""

    user = f"""Write a CEO morning briefing for {COMPANY_NAME}.

KEY INTELLIGENCE:
- Top Opportunity: {top_opp}
- Top Risk: {top_risk}
- Market Sentiment: {sentiment} ({pos_pct}% positive, {neg_pct}% negative)
- Top Recommendation: {top_rec}

Structure your briefing with these three sections:

WHAT HAPPENED
(2-3 bullet points on the most important recent developments)

WHY IT MATTERS
(2-3 bullet points explaining business implications)

WHAT TO DO NEXT
(3-4 prioritized action items for this week)

Keep the entire briefing under 300 words. Be specific, not generic."""

    briefing = _llm(system, user, max_tokens=600)
    return briefing


# ──────────────────────────────────────────
#  Answer custom CEO question (RAG Q&A)
# ──────────────────────────────────────────
def answer_ceo_question(question: str) -> str:
    """
    Let the CEO ask any strategic question and get a RAG-grounded answer.

    Example questions:
      - "What are competitors doing in AI?"
      - "Should we expand into automotive AI?"
      - "What is the market saying about our H100?"
    """
    # Retrieve relevant evidence
    chunks = kb_query(question, top_k=TOP_K_RETRIEVAL)
    context = "\n\n".join([f"[{c['source']} | {c['date']}]\n{c['text']}" for c in chunks])

    system = f"""You are a strategic advisor to the CEO of {COMPANY_NAME}.
Answer questions with evidence-based reasoning. Be concise and executive in tone.
Always cite the sources you use."""

    user = f"""CONTEXT:
{context}

CEO QUESTION: {question}

Provide a direct, evidence-based answer in 150-200 words. 
End with: "Based on evidence from: [list source names]" """

    return _llm(system, user)


if __name__ == "__main__":
    # Quick test of the Q&A
    ans = answer_ceo_question("What is the biggest risk facing NVIDIA right now?")
    print(ans)
