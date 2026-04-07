"""
test_google_jobs_tool.py — Interactive debug runner for GoogleJobsCollectorTool.

Usage:
    python test_google_jobs_tool.py
    python test_google_jobs_tool.py --titles "AI Engineer" "ML Engineer" --country Germany --limit 5

What it does:
  1. Calls GoogleJobsCollectorTool with your parameters.
  2. Prints a summary of every collected posting.
  3. Saves all intermediate + final data to debug_output/ as JSON files:
       debug_output/raw_postings.json        — full tool output
       debug_output/batch_<n>_input.json     — what requirements_extractor sends to the LLM
       debug_output/batch_<n>_raw_llm.txt    — raw LLM response (before JSON parsing)
       debug_output/batch_<n>_parsed.json    — parsed requirements
       debug_output/all_requirements.json    — combined extraction result

Run requirements:
  - SERPAPI_API_KEY and OPENAI_API_KEY must be set (via .env or environment).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Load .env so API keys are available when running locally.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — keys must be in the environment already

# Setup logging so all INFO/WARNING from the tools and nodes appears in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("debug_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path)


def run_collection(job_titles: list[str], country: str, limit: int) -> list[dict]:
    """Run GoogleJobsCollectorTool and save raw output."""
    from tools.google_jobs_tool import GoogleJobsCollectorTool

    logger.info("=" * 60)
    logger.info("STEP 1 — Collecting job postings via SerpAPI")
    logger.info("  Titles : %s", job_titles)
    logger.info("  Country: %s", country)
    logger.info("  Limit  : %d", limit)
    logger.info("=" * 60)

    tool = GoogleJobsCollectorTool()
    postings = tool.run({"job_titles": job_titles, "country": country, "limit": limit})

    logger.info("Collected %d postings", len(postings))

    # Save full raw output
    save_json(OUTPUT_DIR / "raw_postings.json", postings)

    # Print a human-readable summary
    print("\n--- RAW POSTINGS SUMMARY ---")
    for i, p in enumerate(postings, 1):
        desc_len = len(p.get("description") or "")
        print(f"  [{i:2d}] {p.get('title', '?')} @ {p.get('company', '?')} ({p.get('location', '?')}) — desc {desc_len} chars")

    return postings


def run_extraction(postings: list[dict]) -> list[dict]:
    """Run the requirements_extractor logic and save intermediates per batch."""
    import re

    import requests
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2 — Extracting requirements via LLM (gpt-4o-mini)")
    logger.info("=" * 60)

    _SYSTEM_PROMPT = """You are a job requirements parser.
For each job posting extract:
- technical_skills: programming languages, frameworks, libraries, methodologies (e.g. Python, Docker, Spark)
- cloud_platforms: ONLY from [AWS, Azure, GCP]. Ignore all others.
- certifications: industry certifications mentioned (e.g. AWS Certified, CKA, PMP)

Return a JSON array where each element corresponds to one job posting:
[
  {
    "job_id": "...",
    "title": "...",
    "company": "...",
    "technical_skills": ["Python", "Spark", ...],
    "cloud_platforms": ["AWS", ...],
    "certifications": ["AWS Certified Solutions Architect", ...]
  },
  ...
]

Be concise. Deduplicate within each job. Normalise names (e.g. "python" -> "Python").
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    all_requirements = []
    batch_size = 10
    num_batches = (len(postings) + 9) // 10

    for i in range(0, len(postings), batch_size):
        batch_num = i // batch_size + 1
        batch = postings[i:i + batch_size]

        logger.info("Batch %d/%d — %d jobs", batch_num, num_batches, len(batch))

        batch_input = [
            {
                "job_id": f"job_{i + idx + 1}",  # Short ID — avoids LLM truncating long base64 SerpAPI IDs
                "title": j.get("title", ""),
                "company": j.get("company", ""),
                "description": (j.get("description") or "")[:3000],
            }
            for idx, j in enumerate(batch)
        ]

        # Save the input we're sending to the LLM
        save_json(OUTPUT_DIR / f"batch_{batch_num}_input.json", batch_input)

        batch_text = json.dumps(batch_input, indent=2)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract requirements from these job postings:\n\n{batch_text}"),
        ])

        raw = response.content.strip()

        # Save raw LLM response BEFORE any parsing attempt
        raw_path = OUTPUT_DIR / f"batch_{batch_num}_raw_llm.txt"
        raw_path.write_text(raw, encoding="utf-8")
        logger.info("Saved raw LLM response: %s (%d chars)", raw_path, len(raw))
        logger.info("Raw LLM response starts with: %r", raw[:120])

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            raw = "\n".join(lines[1:end])
            logger.info("Stripped markdown fences — remaining: %d chars", len(raw))

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict) and "jobs" in parsed:
                items = parsed["jobs"]
            else:
                logger.warning("Unexpected JSON structure: %s", type(parsed))
                items = []

            save_json(OUTPUT_DIR / f"batch_{batch_num}_parsed.json", items)
            logger.info("Batch %d parsed successfully — %d items", batch_num, len(items))
            all_requirements.extend(items)

        except json.JSONDecodeError as e:
            logger.error("Batch %d JSON parse failed: %s", batch_num, e)
            logger.error("Raw content (first 500 chars):\n%s", raw[:500])

    save_json(OUTPUT_DIR / "all_requirements.json", all_requirements)
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXTRACTION DONE — %d requirements extracted from %d postings", len(all_requirements), len(postings))
    logger.info("=" * 60)

    # Print skill summary
    if all_requirements:
        from collections import Counter
        all_skills: Counter = Counter()
        for req in all_requirements:
            for skill in (req.get("technical_skills") or []):
                all_skills[skill.strip()] += 1

        print("\n--- TOP SKILLS FOUND ---")
        for skill, count in all_skills.most_common(20):
            pct = count / len(postings) * 100
            print(f"  {skill:<30} {count:3d}x  ({pct:.0f}%)")

        print("\n--- CLOUD PLATFORMS ---")
        cloud: Counter = Counter()
        for req in all_requirements:
            for p in (req.get("cloud_platforms") or []):
                cloud[p] += 1
        for platform in ["AWS", "Azure", "GCP"]:
            pct = cloud[platform] / len(postings) * 100
            print(f"  {platform:<10} {cloud[platform]:3d}x  ({pct:.0f}%)")

    return all_requirements


def main():
    parser = argparse.ArgumentParser(description="Debug GoogleJobsCollectorTool + requirements extraction")
    parser.add_argument("--titles", nargs="+", default=None, help='Job titles, e.g. --titles "AI Engineer" "ML Engineer"')
    parser.add_argument("--country", default=None, help='Country, e.g. --country Germany')
    parser.add_argument("--limit", type=int, default=None, help="Max postings to collect (default: 5)")
    parser.add_argument("--skip-extraction", action="store_true", help="Only collect postings, skip LLM extraction")
    args = parser.parse_args()

    # Interactive prompts if not provided via CLI
    if args.titles is None:
        raw = input("Job titles (comma-separated) [AI Engineer]: ").strip()
        args.titles = [t.strip() for t in raw.split(",")] if raw else ["AI Engineer"]

    if args.country is None:
        raw = input("Country [USA]: ").strip()
        args.country = raw if raw else "USA"

    if args.limit is None:
        raw = input("Max postings [5]: ").strip()
        args.limit = int(raw) if raw.isdigit() else 5

    print(f"\nOutput will be saved to: {OUTPUT_DIR.resolve()}\n")

    # Check required env vars
    missing = [k for k in ("SERPAPI_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        logger.error("Missing required environment variables: %s", missing)
        sys.exit(1)

    postings = run_collection(args.titles, args.country, args.limit)

    if not postings:
        logger.warning("No postings collected — check your SERPAPI_API_KEY and try different titles/country")
        sys.exit(1)

    if not args.skip_extraction:
        run_extraction(postings)

    print(f"\nAll debug files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
