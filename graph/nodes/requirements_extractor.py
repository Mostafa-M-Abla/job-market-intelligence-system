"""
graph/nodes/requirements_extractor.py — LLM-based extraction of skills, cloud
platforms, and certifications from raw job posting descriptions.

ROLE IN THE GRAPH
-----------------
This is the second node in the sequential data pipeline:
  job_collector -> requirements_extractor -> market_analyzer

It takes the list of raw job postings (each with a text description) and asks
the LLM to extract structured data from each one.  The output is a list of
per-job requirement objects that market_analyzer can then aggregate.

WHY EXTRACT FIRST, THEN AGGREGATE?
------------------------------------
We could ask the LLM to analyse all 30 job postings in one go, but that approach
has two problems:
  1. Context window limits — 30 long job descriptions can easily exceed 100k tokens.
  2. Quality — asking the LLM to count frequencies accurately across 30 documents
     in a single pass produces less reliable results than a two-step approach.

By splitting into "extract per-job" (this node) then "aggregate across jobs"
(market_analyzer), we get better accuracy and stay within context limits.

BATCHING
--------
We process postings in batches of 10.  Each batch is a separate LLM call.
This keeps the input size predictable and avoids hitting the model's context limit
even with unusually long job descriptions.

The description for each posting is capped at 3,000 characters to prevent a single
very long posting from dominating the batch.

MODEL CHOICE
------------
gpt-4o-mini is used here (not gpt-4o) because structured extraction is a simpler
task than aggregation.  The mini model is faster and cheaper, and the structured
output schema (list of JSON objects) keeps the response format consistent.

OUTPUTS written to state
------------------------
  extracted_requirements : list of dicts, one per job posting, with keys:
    - job_id, title, company
    - technical_skills (list of strings)
    - cloud_platforms (list: only AWS / Azure / GCP)
    - certifications (list of strings)
"""

import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graph.state import JobMarketState

logger = logging.getLogger(__name__)

# System prompt that tells the LLM exactly what to extract and how to format it.
# The model is instructed to:
#   - Return a JSON array (one object per job)
#   - Restrict cloud_platforms to only the big three
#   - Normalise skill names (e.g. "python3" -> "Python")
#   - Deduplicate within each job
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


class JobRequirements(BaseModel):
    """
    Pydantic model representing the extracted requirements for a single job posting.

    All fields have sensible defaults so that if the LLM omits a field
    (e.g. a job has no certifications), we don't crash on missing keys.
    """
    job_id: str = ""
    title: str = ""
    company: str = ""
    technical_skills: list[str] = Field(default_factory=list)
    cloud_platforms: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """
    Wrapper model used when the LLM returns a JSON object with a "jobs" key
    instead of a bare JSON array.  Both formats are handled in the parser below.
    """
    jobs: list[JobRequirements]


def requirements_extractor(state: JobMarketState) -> dict:
    """
    Extract structured requirements from each raw job posting using an LLM.

    For each batch of (up to) 10 job postings:
      1. Serialize the batch as a JSON string (descriptions capped at 3,000 chars).
      2. Call gpt-4o-mini with the extraction system prompt.
      3. Parse the JSON response into a list of requirement dicts.
      4. Accumulate results across all batches.

    Args:
        state: Current graph state.  Reads raw_job_postings.

    Returns:
        {"extracted_requirements": [list of per-job dicts]}
        Returns an empty list if no postings are available (early exit).

    Notes on error handling:
        If the LLM returns malformed JSON for a batch, we skip that batch
        (json.JSONDecodeError is caught and silently ignored).  This is a
        deliberate trade-off: a partial result is more useful than a crash,
        and market_analyzer can still aggregate whatever it receives.
    """
    postings = state.get("raw_job_postings") or []

    if not postings:
        logger.info("requirements_extractor: no postings — skipping extraction")
        return {"extracted_requirements": []}

    num_batches = (len(postings) + 9) // 10
    logger.info("requirements_extractor: extracting from %d postings (%d batches)", len(postings), num_batches)

    # temperature=0 for deterministic, consistent extraction.
    # We want the same job to produce the same skill list every time.
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    all_requirements = []
    batch_size = 10  # Process 10 postings per LLM call to stay within context limits.

    for i in range(0, len(postings), batch_size):
        batch = postings[i:i + batch_size]
        logger.info("requirements_extractor: processing batch %d/%d (%d jobs)", i // batch_size + 1, num_batches, len(batch))

        # Serialize the batch into a JSON string.
        # We only include the fields the LLM needs — we don't send everything.
        # Descriptions are capped at 3,000 characters to avoid very long inputs.
        # IMPORTANT: We use short sequential IDs (job_1, job_2, ...) instead of
        # the real SerpAPI job_id, which is a very long base64 string.  When the
        # LLM echoes that string back it sometimes truncates it with "...", which
        # produces invalid JSON and causes the entire batch to be skipped.
        batch_text = json.dumps([
            {
                "job_id": f"job_{i + idx + 1}",
                "title": j.get("title", ""),
                "company": j.get("company", ""),
                "description": (j.get("description") or "")[:3000],
            }
            for idx, j in enumerate(batch)
        ], indent=2)

        # Call the LLM with the extraction prompt.
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract requirements from these job postings:\n\n{batch_text}"),
        ])

        # The LLM is instructed to return a JSON array, but models sometimes:
        #   (a) wrap it in {"jobs": [...]}
        #   (b) wrap it in markdown code fences: ```json ... ```
        # We handle all three formats.
        raw = response.content.strip()
        if raw.startswith("```"):
            # Strip the opening fence line (```json or ```) and closing ```
            lines = raw.split("\n")
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            raw = "\n".join(lines[1:end])
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                # Standard format: LLM returned a bare JSON array.
                all_requirements.extend(parsed)
            elif isinstance(parsed, dict) and "jobs" in parsed:
                # Wrapped format: LLM returned {"jobs": [...]}.
                all_requirements.extend(parsed["jobs"])
            logger.info("requirements_extractor: batch %d parsed — %d items", i // batch_size + 1, len(parsed) if isinstance(parsed, list) else len(parsed.get("jobs", [])))
        except json.JSONDecodeError as e:
            logger.warning("requirements_extractor: batch %d returned invalid JSON — skipping. Error: %s. Raw (first 200): %s",
                           i // batch_size + 1, e, raw[:200])

    logger.info("requirements_extractor: done — %d job requirements extracted", len(all_requirements))
    return {"extracted_requirements": all_requirements}
