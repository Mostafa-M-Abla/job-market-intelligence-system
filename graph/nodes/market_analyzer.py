"""
graph/nodes/market_analyzer.py — Aggregate skill data across all job postings.

ROLE IN THE GRAPH
-----------------
This is the third node in the sequential data pipeline:
  job_collector -> requirements_extractor -> market_analyzer

It receives the per-job extracted requirements (from requirements_extractor)
and asks the LLM to aggregate them into a comprehensive market analysis report.

After producing the report it immediately writes the results to the DB cache
so future requests with the same parameters can skip this entire pipeline.

WHY GPT-4O (not gpt-4o-mini)?
------------------------------
This node performs nuanced reasoning: deduplicating synonyms ("Python" vs
"python3"), counting frequencies accurately, and writing insightful analysis.
The stronger model produces significantly better quality here.

OUTPUTS written to state
------------------------
  market_analysis_markdown : a markdown report with skill/platform/cert tables
                             and key insights
"""

import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import DISABLE_CACHE
from graph.state import JobMarketState
from tools.market_cache_tool import MarketCacheWriteTool

logger = logging.getLogger(__name__)

# System prompt that instructs the LLM how to aggregate the extracted data.
# The {placeholders} are filled in at runtime with the actual job titles,
# country, and post count so the report header is accurate.
_SYSTEM_PROMPT = """You are a job market analyst.
Still 
You will receive a JSON array of extracted job requirements (skills, cloud platforms, certifications).
Aggregate and analyse them to produce a comprehensive market analysis report in Markdown.

The report must include:

## Market Analysis: {job_titles} in {country}
*Based on {total} job postings*

### Top Technical Skills
| Skill | Frequency | % of Jobs |
|---|---|---|
(List top 20 skills sorted by frequency, show count and percentage)

### Cloud Platforms
| Platform | Frequency | % of Jobs |
|---|---|---|
(Only AWS, Azure, GCP — show all three even if frequency is 0)

### Top Certifications
| Certification | Frequency | % of Jobs |
|---|---|---|
(Top 5 certifications)

### Key Insights
(3-5 bullet points highlighting notable patterns, trends, or observations)

Be accurate with percentages. Round to 1 decimal place."""


def market_analyzer(state: JobMarketState) -> dict:
    """
    Aggregate per-job requirements into a market analysis report and cache it.

    Steps:
      1. Fill in the system prompt template with actual values.
      2. Pass the extracted requirements to GPT-4o for aggregation.
      3. Write the results to the DB cache so they can be reused.

    Args:
        state: Current graph state.  Reads extracted_requirements, job_titles,
               country, raw_job_postings, and cache_key.

    Returns:
        {"market_analysis_markdown": "<full markdown report string>"}
    """
    requirements = state.get("extracted_requirements") or []
    job_titles = state.get("job_titles") or []
    country = state.get("country") or "Unknown"
    # Use the raw postings count as the denominator for percentage calculations.
    total = len(state.get("raw_job_postings") or [])

    logger.info(
        "market_analyzer: aggregating %d requirements from %d postings for %s in %s",
        len(requirements), total, job_titles, country,
    )

    # GPT-4o for better quality aggregation (synonym deduplication, accurate counts).
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,   # Deterministic — we want consistent frequency counts
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Inject the actual values into the prompt template.
    system = _SYSTEM_PROMPT.format(
        job_titles=", ".join(job_titles),
        country=country,
        total=total,
    )

    logger.info("market_analyzer: calling LLM (gpt-4o) for aggregation")
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Here are the extracted requirements:\n\n{json.dumps(requirements, indent=2)}"),
    ])
    logger.info("market_analyzer: LLM done — %d chars of analysis generated", len(response.content))

    analysis_md = response.content

    cache_key = state.get("cache_key")
    if cache_key and total > 0 and not DISABLE_CACHE:
        logger.info("market_analyzer: writing result to cache (key %s…)", cache_key[:12])
        write_tool = MarketCacheWriteTool()
        write_tool.run({
            "cache_key": cache_key,
            "job_titles": job_titles,
            "country": country,
            "raw_job_postings": state.get("raw_job_postings") or [],
            "extracted_requirements": requirements,
            "market_analysis_markdown": analysis_md,
            "total_posts": total,
        })

    elif not (cache_key and total > 0 and not DISABLE_CACHE):
        logger.info("market_analyzer: skipping cache write (total=%d, disable_cache=%s)", total, DISABLE_CACHE)

    return {"market_analysis_markdown": analysis_md}
