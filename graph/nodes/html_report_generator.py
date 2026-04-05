"""
html_report_generator node — generates a full, styled HTML report and saves it to disk.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import JobMarketState
from tools.html_report_saver import HTMLReportSaverTool

_SYSTEM_PROMPT = """You are an expert data visualization specialist who creates beautiful HTML reports.

Generate a complete, self-contained HTML report. Requirements:
- Include all CSS inline in a <style> block (no external dependencies)
- Professional color scheme (blues and grays work well)
- Responsive layout
- Sections: Executive Summary, Market Overview, Top Skills (with visual bars),
  Cloud Platforms, Certifications, and (if provided) Resume Skill Gap & Recommendations
- Use progress bars or simple visual indicators for skill frequencies
- Include a header with the job titles, country, and date
- Footer with a note that this is AI-generated analysis

Return ONLY the raw HTML. Do not wrap in markdown code blocks."""


def html_report_generator(state: JobMarketState) -> dict:
    market_analysis = state.get("market_analysis_markdown") or ""
    skill_gap = state.get("skill_gap_markdown") or ""
    job_titles = state.get("job_titles") or []
    country = state.get("country") or ""
    session_id = state.get("session_id", "default")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    content_parts = []
    if market_analysis:
        content_parts.append(f"## Market Analysis\n\n{market_analysis}")
    if skill_gap:
        content_parts.append(f"## Skill Gap & Recommendations\n\n{skill_gap}")

    context = "\n\n---\n\n".join(content_parts)

    prompt = (
        f"Generate a complete HTML report for:\n"
        f"Job Titles: {', '.join(job_titles)}\n"
        f"Country: {country}\n\n"
        f"Use this analysis data:\n\n{context}"
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    html = response.content.strip()
    # Strip markdown code fences if the model included them
    if html.startswith("```"):
        lines = html.split("\n")
        html = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    saver = HTMLReportSaverTool()
    report_url = saver.run({"html_text": html, "session_id": session_id})

    return {"html_report_path": report_url}
