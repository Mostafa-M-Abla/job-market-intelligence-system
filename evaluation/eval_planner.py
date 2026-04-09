"""
evaluation/eval_planner.py

Eval 2: Planner Node Isolation
Tests the planner node in isolation — no graph execution, no SerpAPI, no DB.
Evaluates task decomposition quality: count, intent, anti-over-decomposition,
job title extraction, and country extraction.

Dataset: eval-planner-v1 (stored in LangSmith)
Usage:   python evaluation/eval_planner.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

from evaluation.shared.dataset_utils import get_or_create_dataset, upsert_examples
from graph.nodes.planner import planner

DATASET_NAME = "eval-planner-v1"
DATASET_DESCRIPTION = (
    "Planner node isolation eval. 8 examples covering: single-task queries, "
    "multi-task queries, anti-over-decomposition cases, and all 4 intent types. "
    "Zero SerpAPI calls — only OpenAI gpt-4o-mini."
)

EXAMPLES = [
    # ── P1: Single general_question — anti-over-decompose ────────────────────
    {
        "inputs": {"message": "What is the difference between LangGraph and CrewAI?"},
        "outputs": {
            "expected_task_count": 1,
            "expected_intents": ["general_question"],
            "expected_job_titles_non_empty": False,
            "expected_country": None,
        },
        "metadata": {
            "example_id": "planner-p1-single-general",
            "note": "Single general knowledge question — must be exactly 1 task",
        },
    },
    # ── P2: Single full_market_analysis ──────────────────────────────────────
    {
        "inputs": {"message": "Do a full market analysis for AI Engineer roles in Germany"},
        "outputs": {
            "expected_task_count": 1,
            "expected_intents": ["full_market_analysis"],
            "expected_job_titles_non_empty": True,
            "expected_country": "Germany",
        },
        "metadata": {"example_id": "planner-p2-single-full-market"},
    },
    # ── P3: Single focused_question ───────────────────────────────────────────
    {
        "inputs": {
            "message": "What cloud platforms are most in demand for AI Engineer jobs in Germany?"
        },
        "outputs": {
            "expected_task_count": 1,
            "expected_intents": ["focused_question"],
            "expected_job_titles_non_empty": True,
            "expected_country": "Germany",
        },
        "metadata": {"example_id": "planner-p3-single-focused"},
    },
    # ── P4: Single resume_analysis ────────────────────────────────────────────
    {
        "inputs": {
            "message": "Compare my resume against the AI Engineer job market in Germany"
        },
        "outputs": {
            "expected_task_count": 1,
            "expected_intents": ["resume_analysis"],
            "expected_job_titles_non_empty": True,
            "expected_country": "Germany",
        },
        "metadata": {"example_id": "planner-p4-single-resume"},
    },
    # ── P5: Anti-over-decomposition — same market, two job titles ─────────────
    # "What skills do AI and ML engineers need in Germany?" is ONE market query.
    # The planner must NOT split this into 2 tasks.
    {
        "inputs": {
            "message": "What skills do AI engineers and ML engineers need in Germany?"
        },
        "outputs": {
            "expected_task_count": 1,
            "expected_intents": ["focused_question"],
            "expected_job_titles_non_empty": True,
            "expected_country": "Germany",
        },
        "metadata": {
            "example_id": "planner-p5-anti-overdecompose-same-market",
            "note": "Same market — must stay 1 task despite 2 job titles",
        },
    },
    # ── P6: Anti-over-decomposition — simple general knowledge ────────────────
    {
        "inputs": {"message": "Explain what Docker is and why developers use it"},
        "outputs": {
            "expected_task_count": 1,
            "expected_intents": ["general_question"],
            "expected_job_titles_non_empty": False,
            "expected_country": None,
        },
        "metadata": {
            "example_id": "planner-p6-anti-overdecompose-general",
            "note": "Single knowledge question — must be exactly 1 task",
        },
    },
    # ── P7: Multi-task — 2 distinct markets ──────────────────────────────────
    {
        "inputs": {
            "message": (
                "Analyse AI Engineer jobs in Germany AND Data Engineer jobs in France"
            )
        },
        "outputs": {
            "expected_task_count": 2,
            "expected_intents": ["full_market_analysis", "full_market_analysis"],
            "expected_job_titles_non_empty": True,
            "expected_country": None,  # N/A at top level — each task has its own
        },
        "metadata": {
            "example_id": "planner-p7-multi-two-markets",
            "note": "Two distinct markets → 2 tasks",
        },
    },
    # ── P8: Multi-task — general + focused ───────────────────────────────────
    {
        "inputs": {
            "message": (
                "Explain what MLOps is, and also tell me what tools are "
                "most used by ML Engineers in the UK"
            )
        },
        "outputs": {
            "expected_task_count": 2,
            "expected_intents": ["general_question", "focused_question"],
            "expected_job_titles_non_empty": True,
            "expected_country": None,
        },
        "metadata": {
            "example_id": "planner-p8-multi-general-plus-focused",
            "note": "General knowledge + market data question → 2 tasks",
        },
    },
]


# ── Target function ───────────────────────────────────────────────────────────

def _call_planner(message: str) -> list[dict]:
    """Call the planner node directly with a minimal state dict."""
    state = {
        "messages": [HumanMessage(content=message)],
        "session_id": "eval-planner",
    }
    return planner(state)["task_queue"]


def target(inputs: dict) -> dict:
    """
    Call the planner node in isolation.

    Args:
        inputs: {"message": str}

    Returns:
        {
            "task_count": int,
            "actual_intents": list[str],
            "tasks": list[dict],
        }
    """
    tasks = _call_planner(inputs["message"])
    return {
        "task_count": len(tasks),
        "actual_intents": [t.get("intent", "") for t in tasks],
        "tasks": tasks,
    }


# ── Evaluators (all deterministic, zero LLM calls) ───────────────────────────

def eval_task_count_correct(run: Run, example: Example) -> EvaluationResult:
    """Exact match on number of tasks produced."""
    expected = example.outputs["expected_task_count"]
    actual = run.outputs.get("task_count", 0)
    return EvaluationResult(
        key="task_count_correct",
        score=1.0 if actual == expected else 0.0,
        comment=f"Expected {expected} task(s), got {actual}",
    )


def eval_intent_correct(run: Run, example: Example) -> EvaluationResult:
    """All tasks must have the correct intent (order-insensitive for multi-task)."""
    expected = sorted(example.outputs["expected_intents"])
    actual = sorted(run.outputs.get("actual_intents", []))
    match = actual == expected
    return EvaluationResult(
        key="intent_correct",
        score=1.0 if match else 0.0,
        comment=f"Expected {expected}, got {actual}",
    )


def eval_no_over_decomposition(run: Run, example: Example) -> EvaluationResult:
    """
    For single-task examples: score 1.0 if exactly 1 task produced, else 0.0.
    For multi-task examples: always 1.0 (over-decomposition only matters for 1-task queries).
    """
    expected = example.outputs["expected_task_count"]
    actual = run.outputs.get("task_count", 0)
    if expected > 1:
        return EvaluationResult(key="no_over_decomposition", score=1.0, comment="N/A for multi-task")
    score = 1.0 if actual == 1 else 0.0
    return EvaluationResult(
        key="no_over_decomposition",
        score=score,
        comment=f"Expected 1 task, got {actual}",
    )


def eval_job_titles_extracted(run: Run, example: Example) -> EvaluationResult:
    """
    When job titles are expected, check that at least one task has a non-empty
    job_titles list. For general_question, titles are not expected — score 1.0.
    """
    expect_titles = example.outputs.get("expected_job_titles_non_empty", False)
    tasks = run.outputs.get("tasks", [])

    if not expect_titles:
        return EvaluationResult(key="job_titles_extracted", score=1.0, comment="Titles not expected for this intent")

    has_titles = any(bool(t.get("job_titles")) for t in tasks)
    return EvaluationResult(
        key="job_titles_extracted",
        score=1.0 if has_titles else 0.0,
        comment="Job titles present" if has_titles else "No job titles extracted in any task",
    )


def eval_country_extracted(run: Run, example: Example) -> EvaluationResult:
    """
    When a specific country is expected, check that at least one task has it.
    Case-insensitive comparison. If no country expected, always score 1.0.
    """
    expected_country = example.outputs.get("expected_country")
    if expected_country is None:
        return EvaluationResult(key="country_extracted", score=1.0, comment="Country not expected")

    tasks = run.outputs.get("tasks", [])
    found = any(
        (t.get("country") or "").lower() == expected_country.lower()
        for t in tasks
    )
    actual_countries = [t.get("country") for t in tasks]
    return EvaluationResult(
        key="country_extracted",
        score=1.0 if found else 0.0,
        comment=f"Expected '{expected_country}', tasks had: {actual_countries}",
    )


# ── Orchestration ─────────────────────────────────────────────────────────────

def setup_dataset(client: Client) -> str:
    print(f"  Setting up dataset: {DATASET_NAME}")
    ds = get_or_create_dataset(client, DATASET_NAME, DATASET_DESCRIPTION)
    upsert_examples(client, ds.id, EXAMPLES)
    return DATASET_NAME


def run_eval(client: Client) -> None:
    dataset_name = setup_dataset(client)
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            eval_task_count_correct,
            eval_intent_correct,
            eval_no_over_decomposition,
            eval_job_titles_extracted,
            eval_country_extracted,
        ],
        experiment_prefix="planner-node-isolation",
        metadata={"eval_version": "1", "node": "planner", "model": "gpt-4o-mini"},
        client=client,
    )
    print("  Eval 2 complete.")
    try:
        print(f"  Results: {results.experiment_results_url}")
    except Exception:
        pass


if __name__ == "__main__":
    client = Client()
    run_eval(client)
