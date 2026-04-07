"""
graph/nodes/skill_gap_analyzer.py — Personalised skill gap analysis by comparing
the user's resume against the current job market requirements.

ROLE IN THE GRAPH
-----------------
This node runs only for the resume_analysis intent, after we have both:
  1. The user's resume text (from resume_parser)
  2. The market analysis (from market_analyzer or cache_lookup)

Flow position:
  [market data ready] -> skill_gap_analyzer -> confirm_report_format -> ...

The node asks GPT-4o to act as a career development specialist: read the resume,
read the market analysis, and produce a structured Markdown report that tells the
user exactly which skills they are missing and what to learn next.

WHY GPT-4O (not mini)?
-----------------------
This node requires nuanced reasoning:
  - Inferring the user's skills from free-form resume text
  - Matching those skills against market frequencies
  - Prioritising the skill gaps by impact
  - Writing actionable, personalised learning path recommendations

The larger model produces significantly more useful output for this task.

RESUME TEXT LIMIT
-----------------
The resume text is capped at 4,000 characters before being sent to the LLM.
This keeps token costs predictable and stays well within the model's context
window even when combined with a long market analysis report.

OUTPUTS written to state
------------------------
  skill_gap_markdown : a Markdown string containing:
    - "Your Current Skills" section
    - A table of top 15 market skills with Yes/No presence in the resume
    - "Top 5 Skills to Learn Next" with learning paths
    - "Certifications to Consider" section
    - A personalised 2-3 sentence summary
"""

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import JobMarketState

logger = logging.getLogger(__name__)

# Detailed system prompt defining the output structure.
# Using a fixed template ensures consistent, comparable output across sessions.
_SYSTEM_PROMPT = """You are a career development specialist.

You will receive:
1. The user's resume text (or extracted skills)
2. A job market analysis report

Your task:
1. Extract the user's current technical skills, cloud platforms, and certifications from the resume.
2. Compare them against the market analysis.
3. Identify the most impactful skills the user is missing.

Produce a structured Markdown report:

## Resume vs Market Analysis

### Your Current Skills
(Brief summary of what the user already has)

### Skills Gap Analysis
| Skill | Market Demand | You Have It? | Priority |
|---|---|---|---|
(Top 15 skills from the market — mark Yes/No for user, priority: High/Medium/Low)

### Top 5 Skills to Learn Next
For each skill:
**1. [Skill Name]** — Appears in X% of job postings
- Why it matters: ...
- Suggested learning path: ...

### Certifications to Consider
(Top 2-3 certifications you don't have that appear frequently)

### Summary
(2-3 sentence personalised summary)

Be specific and actionable. Base all percentages on the market analysis data."""


def skill_gap_analyzer(state: JobMarketState) -> dict:
    """
    Compare the user's resume against the market analysis to identify skill gaps.

    Steps:
      1. Read resume_text and market_analysis_markdown from state.
      2. If either is missing, return a fallback message — the analysis can't
         be personalised without both inputs.
      3. Call GPT-4o with the system prompt and the combined context.
      4. Store the resulting Markdown in state["skill_gap_markdown"].

    Args:
        state: Current graph state.  Reads resume_text and market_analysis_markdown.

    Returns:
        {"skill_gap_markdown": "<structured Markdown analysis>"}
        Returns a brief error message string if inputs are insufficient.

    The skill_gap_markdown field is later used by:
      - html_report_generator: includes it as a "Skill Gap & Recommendations" section
      - answer_focused: includes it as extra context when intent is resume_analysis
      - respond: references it in the "report is ready" message to confirm personalisation
    """
    resume_text = state.get("resume_text") or ""
    market_analysis = state.get("market_analysis_markdown") or ""

    # Both inputs are required for a meaningful comparison.
    # If either is missing, return a safe fallback rather than crashing.
    if not resume_text or not market_analysis:
        logger.warning("skill_gap_analyzer: missing resume_text=%s market_analysis=%s — skipping",
                       bool(resume_text), bool(market_analysis))
        return {"skill_gap_markdown": "Insufficient data for skill gap analysis."}

    # temperature=0 for consistent, reproducible analysis.
    # We want the same resume + market data to produce the same recommendations.
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Combine resume and market analysis into a single context block.
    # The resume is capped at 4,000 characters to keep token costs predictable.
    # (Most resumes are 1-3 pages / ~1,500-3,000 chars, so this rarely truncates.)
    user_content = (
        f"## Resume\n\n{resume_text[:4000]}\n\n"
        f"## Market Analysis\n\n{market_analysis}"
    )

    logger.info("skill_gap_analyzer: calling LLM (gpt-4o) — resume %d chars, market analysis %d chars",
                len(resume_text), len(market_analysis))
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    logger.info("skill_gap_analyzer: done — %d chars of skill gap analysis", len(response.content))
    return {"skill_gap_markdown": response.content}
