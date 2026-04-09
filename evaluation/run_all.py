"""
evaluation/run_all.py

Runs all three LangSmith evaluations in sequence.

Order (cheapest first to fail fast):
  1. Eval 2 — Planner node isolation (OpenAI only, no SerpAPI)
  2. Eval 3 — Trajectory (SerpAPI for 1 of 3 examples)
  3. Eval 1 — Final answer quality (SerpAPI for 2 of 5 examples)

Usage: python evaluation/run_all.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

import evaluation.eval_final_answer as eval1
import evaluation.eval_planner as eval2
import evaluation.eval_trajectory as eval3


def main():
    client = Client()
    print("=" * 60)
    print("LangSmith Evaluation Suite — Job Market Intelligence")
    print(f"Project: {os.getenv('LANGSMITH_PROJECT', 'job-market-intelligence-system')}")
    print("=" * 60)

    print("\n[1/3] Eval 2: Planner Node Isolation (no SerpAPI)")
    eval2.run_eval(client)

    print("\n[2/3] Eval 3: Trajectory")
    eval3.run_eval(client)

    print("\n[3/3] Eval 1: Final Answer Quality")
    eval1.run_eval(client)

    print("\n" + "=" * 60)
    print("All evaluations complete.")
    print("View results in LangSmith under:")
    print(f"  Project: {os.getenv('LANGSMITH_PROJECT', 'job-market-intelligence-system')}")
    print("  Experiments: planner-node-isolation, trajectory-node-sequence, final-answer-quality")
    print("=" * 60)


if __name__ == "__main__":
    main()
