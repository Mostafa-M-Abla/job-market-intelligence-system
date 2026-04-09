"""
evaluation/eval_final_answer.py

Eval 1: Final Answer Quality
Runs the full graph end-to-end for 5 questions and scores the responses
with an LLM-as-a-judge on relevance, correctness, and completeness.

Dataset: eval-final-answer-v1 (stored in LangSmith)
Usage:   python evaluation/eval_final_answer.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate

from evaluation.shared.dataset_utils import get_or_create_dataset, upsert_examples
from evaluation.shared.graph_runner import run_graph_with_hitl
from evaluation.shared.llm_judge import (
    evaluate_completeness,
    evaluate_correctness,
    evaluate_relevance,
)

DATASET_NAME = "eval-final-answer-v1"
DATASET_DESCRIPTION = (
    "End-to-end answer quality evaluation. "
    "5 questions: 3 general_question (no SerpAPI/HITL), "
    "2 focused_question (SerpAPI + auto-confirmed HITL)."
)

EXAMPLES = [
    # ── Q1: general_question ─────────────────────────────────────────────────
    {
        "inputs": {
            "message": "What is the difference between LangGraph and LangChain?",
        },
        "outputs": {
            "reference_answer": (
                "LangChain is a framework for building LLM-powered applications "
                "through composable chains and tool integrations. LangGraph extends "
                "LangChain to support stateful, multi-actor workflows modelled as "
                "directed graphs with cycles, enabling persistent state, human-in-the-loop "
                "pauses, and complex agent coordination. LangChain suits linear pipelines; "
                "LangGraph suits agents that need to loop, branch, and resume."
            ),
        },
        "metadata": {"example_id": "final-q1-langgraph-vs-langchain"},
    },
    # ── Q2: general_question ─────────────────────────────────────────────────
    {
        "inputs": {
            "message": "What are the most important skills for a machine learning engineer in 2025?",
        },
        "outputs": {
            "reference_answer": (
                "Core skills include: Python proficiency, deep learning frameworks (PyTorch, "
                "TensorFlow), MLOps tooling (MLflow, Kubeflow, SageMaker), cloud platforms "
                "(AWS, GCP, Azure), data engineering (Spark, SQL), and model evaluation. "
                "Increasingly important: LLM fine-tuning, RAG architecture, vector databases, "
                "and responsible AI practices."
            ),
        },
        "metadata": {"example_id": "final-q2-ml-engineer-skills"},
    },
    # ── Q3: general_question ─────────────────────────────────────────────────
    {
        "inputs": {
            "message": "Explain what RAG (Retrieval-Augmented Generation) is and when to use it.",
        },
        "outputs": {
            "reference_answer": (
                "RAG is a technique that combines a retrieval system (typically a vector "
                "database) with a generative LLM. The retrieval step finds relevant documents "
                "for the query; the generation step uses those documents as context to produce "
                "a grounded answer. Use RAG when: (1) the LLM's training data is outdated, "
                "(2) you need citations or traceability, (3) you have proprietary knowledge the "
                "model was not trained on, or (4) hallucination on factual questions is unacceptable."
            ),
        },
        "metadata": {"example_id": "final-q3-rag-explanation"},
    },
    # ── Q4: focused_question — SerpAPI + 1 HITL ──────────────────────────────
    {
        "inputs": {
            "message": "What cloud platforms are most in demand for AI Engineer jobs in Germany?",
        },
        "outputs": {
            "reference_answer": (
                "AWS, Google Cloud (GCP), and Azure are consistently the most in-demand "
                "cloud platforms for AI Engineer roles in Germany. AWS typically leads, "
                "followed by GCP (popular for ML/AI workloads due to Vertex AI and BigQuery) "
                "and Azure (common in enterprise environments). Some postings also mention "
                "on-premise solutions or hybrid cloud."
            ),
        },
        "metadata": {"example_id": "final-q4-cloud-platforms-germany"},
    },
    # ── Q5: focused_question — SerpAPI + 1 HITL ──────────────────────────────
    {
        "inputs": {
            "message": "What programming languages are required for Data Engineer roles in the UK?",
        },
        "outputs": {
            "reference_answer": (
                "Python is overwhelmingly the most required language for Data Engineer roles "
                "in the UK, appearing in the majority of job postings. SQL is universally "
                "expected. Scala is commonly required for Spark-heavy roles. Java appears "
                "less frequently. Some postings mention R for analytics-adjacent roles."
            ),
        },
        "metadata": {"example_id": "final-q5-data-engineer-languages-uk"},
    },
]


def target(inputs: dict) -> dict:
    """
    Run the full graph for a single question and return the final answer.

    Args:
        inputs: {"message": str}

    Returns:
        {"final_answer": str}
    """
    result = run_graph_with_hitl(message=inputs["message"], total_posts=3)
    messages = result.get("messages", [])
    final_answer = ""
    if messages:
        last = messages[-1]
        final_answer = last.content if hasattr(last, "content") else str(last)
    return {"final_answer": final_answer}


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
        evaluators=[evaluate_relevance, evaluate_correctness, evaluate_completeness],
        experiment_prefix="final-answer-quality",
        metadata={"eval_version": "1", "judge_model": "gpt-4o-mini"},
        client=client,
    )
    print(f"  Eval 1 complete.")
    try:
        print(f"  Results: {results.experiment_results_url}")
    except Exception:
        pass


if __name__ == "__main__":
    client = Client()
    run_eval(client)
