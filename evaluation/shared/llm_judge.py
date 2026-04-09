"""
evaluation/shared/llm_judge.py

LLM-as-a-judge evaluators for Eval 1 (final answer quality).

Three evaluator functions — evaluate_relevance, evaluate_correctness,
evaluate_completeness — each scoring 0.0–1.0 using gpt-4o-mini with
structured output.

LangSmith calls each evaluator independently with (run, example), so each
makes its own LLM call. This is correct and keeps the evaluators decoupled.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field


_JUDGE_SYSTEM = """You are an expert evaluator of AI-generated answers about job markets, \
technology careers, and software engineering concepts.

You will be given a QUESTION, a REFERENCE ANSWER (ground truth), and an ANSWER TO EVALUATE.

Score the answer on THREE dimensions, each from 0.0 to 1.0:
- relevance:    Does the answer directly address what the question asked?
                1.0 = fully on-topic and addresses the question directly
                0.0 = completely off-topic or doesn't answer the question
- correctness:  Are the facts, claims, and recommendations accurate and reasonable?
                1.0 = fully accurate, no misleading claims
                0.0 = factually wrong or highly misleading
- completeness: Does the answer cover all important aspects of the question?
                1.0 = comprehensive, covers all key points from the reference
                0.0 = misses most key aspects

Return a JSON object with keys: relevance, correctness, completeness, reasoning.
reasoning should be 1-2 sentences explaining the scores."""


class JudgeScores(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _call_judge(question: str, reference_answer: str, actual_answer: str) -> JudgeScores:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).with_structured_output(JudgeScores)

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"ANSWER TO EVALUATE:\n{actual_answer}"
    )
    return llm.invoke([SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)])


def evaluate_relevance(run: Run, example: Example) -> EvaluationResult:
    scores = _call_judge(
        question=example.inputs["message"],
        reference_answer=example.outputs.get("reference_answer", ""),
        actual_answer=run.outputs.get("final_answer", ""),
    )
    return EvaluationResult(key="relevance", score=scores.relevance, comment=scores.reasoning)


def evaluate_correctness(run: Run, example: Example) -> EvaluationResult:
    scores = _call_judge(
        question=example.inputs["message"],
        reference_answer=example.outputs.get("reference_answer", ""),
        actual_answer=run.outputs.get("final_answer", ""),
    )
    return EvaluationResult(key="correctness", score=scores.correctness, comment=scores.reasoning)


def evaluate_completeness(run: Run, example: Example) -> EvaluationResult:
    scores = _call_judge(
        question=example.inputs["message"],
        reference_answer=example.outputs.get("reference_answer", ""),
        actual_answer=run.outputs.get("final_answer", ""),
    )
    return EvaluationResult(key="completeness", score=scores.completeness, comment=scores.reasoning)
