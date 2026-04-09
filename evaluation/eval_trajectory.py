"""
evaluation/eval_trajectory.py

Eval 3: Trajectory Evaluation
Verifies that the correct sequence of graph nodes fires for each query type.
Uses graph.stream(stream_mode="updates") to capture node execution order.

Dataset: eval-trajectory-v1 (stored in LangSmith)
Usage:   python evaluation/eval_trajectory.py

Trajectory note for T2 (focused_question with HITL):
  confirm_search_params executes fully before interrupt() pauses the stream,
  so it appears in the trajectory from the first stream call.
  Post-resume nodes (job_collector onward) appear in the second stream call.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

from evaluation.shared.dataset_utils import get_or_create_dataset, upsert_examples
from evaluation.shared.graph_runner import run_graph_capture_trajectory

DATASET_NAME = "eval-trajectory-v1"
DATASET_DESCRIPTION = (
    "Trajectory evaluation: verifies the correct node sequence fires for each "
    "intent type. 3 examples: simple general_question (no HITL), "
    "focused_question (SerpAPI + auto-confirmed HITL), "
    "compound general+general (2-task loop, no HITL)."
)

EXAMPLES = [
    # ── T1: Simple general_question — no SerpAPI, no HITL ────────────────────
    {
        "inputs": {
            "message": "What is the difference between supervised and unsupervised learning?",
        },
        "outputs": {
            "expected_trajectory": [
                "planner",
                "task_dispatcher",
                "answer_general",
                "respond",
            ],
        },
        "metadata": {
            "example_id": "traj-t1-simple-general",
            "note": "4-node path, no HITL, no loops",
        },
    },
    # ── T2: focused_question — cache miss (DISABLE_CACHE=True), auto-HITL ────
    # confirm_search_params appears because it executes before interrupt() pauses.
    # After resume: job_collector → requirements_extractor → market_analyzer →
    # answer_focused → respond.
    {
        "inputs": {
            "message": "What programming frameworks are most common for AI Engineer jobs in Germany?",
        },
        "outputs": {
            "expected_trajectory": [
                "planner",
                "task_dispatcher",
                "cache_lookup",
                "confirm_search_params",
                "job_collector",
                "requirements_extractor",
                "market_analyzer",
                "answer_focused",
                "respond",
            ],
        },
        "metadata": {
            "example_id": "traj-t2-focused-with-hitl",
            "note": (
                "9-node path. DISABLE_CACHE=True guarantees full path. "
                "confirm_search_params fires before interrupt exits the stream."
            ),
        },
    },
    # ── T3: Compound general+general — 2-task loop, no HITL ──────────────────
    # task_dispatcher and answer_general each fire twice (once per task).
    {
        "inputs": {
            "message": (
                "Explain what a transformer architecture is, "
                "and also explain what transfer learning means"
            ),
        },
        "outputs": {
            "expected_trajectory": [
                "planner",
                "task_dispatcher",
                "answer_general",
                "task_dispatcher",
                "answer_general",
                "respond",
            ],
        },
        "metadata": {
            "example_id": "traj-t3-compound-general-general",
            "note": "6-node path. task_dispatcher and answer_general each fire twice.",
        },
    },
]


# ── Target function ───────────────────────────────────────────────────────────

def target(inputs: dict) -> dict:
    """
    Run the graph and capture the node execution trajectory.

    Args:
        inputs: {"message": str}

    Returns:
        {"trajectory": list[str]}
    """
    _state, trajectory = run_graph_capture_trajectory(
        message=inputs["message"],
        total_posts=3,
    )
    return {"trajectory": trajectory}


# ── Evaluators ────────────────────────────────────────────────────────────────

def eval_trajectory_exact_match(run: Run, example: Example) -> EvaluationResult:
    """Full sequence must match exactly (length + every element in order)."""
    expected = example.outputs["expected_trajectory"]
    actual = run.outputs.get("trajectory", [])
    match = actual == expected
    return EvaluationResult(
        key="trajectory_exact_match",
        score=1.0 if match else 0.0,
        comment=f"Expected: {expected}\nActual:   {actual}",
    )


def eval_trajectory_prefix_match(run: Run, example: Example) -> EvaluationResult:
    """
    Partial credit: score = (expected nodes found in order) / len(expected).

    Uses a greedy subsequence check — all expected nodes must appear in the
    actual trajectory in the same relative order, but extra nodes in the actual
    trajectory are allowed (e.g. repeated nodes in multi-task loops).

    Score 1.0 if all expected nodes appear in order.
    Score 0.5 if half appear.
    Score 0.0 if none appear.
    """
    expected = example.outputs["expected_trajectory"]
    actual = run.outputs.get("trajectory", [])

    actual_idx = 0
    matched = 0
    for exp_node in expected:
        while actual_idx < len(actual):
            if actual[actual_idx] == exp_node:
                matched += 1
                actual_idx += 1
                break
            actual_idx += 1

    score = matched / len(expected) if expected else 0.0
    return EvaluationResult(
        key="trajectory_prefix_match",
        score=score,
        comment=f"Matched {matched}/{len(expected)} expected nodes in order",
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
        evaluators=[eval_trajectory_exact_match, eval_trajectory_prefix_match],
        experiment_prefix="trajectory-node-sequence",
        metadata={"eval_version": "1"},
        client=client,
    )
    print("  Eval 3 complete.")
    try:
        print(f"  Results: {results.experiment_results_url}")
    except Exception:
        pass


if __name__ == "__main__":
    client = Client()
    run_eval(client)
