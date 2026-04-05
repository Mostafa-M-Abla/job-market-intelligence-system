import os
import dotenv
from langsmith import traceable, Client
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

"""
evaluate_html_report.py

This script evaluates the quality of a generated HTML report produced by the multi-agent job market analysis system.
It uses a GPT-4 model via LangChain to score the report on several key criteria:
- Relevance
- Accuracy
- Completeness
- Clarity
- Visual Appeal
- Insights

Each is scored from 1 to 10, and a final_score is provided out of 10.

The evaluation is fully traceable via LangSmith, allowing inspection of prompt inputs, outputs, and LLM reasoning.

✅ Example output:
{
    "relevance": 9,
    "accuracy": 8,
    "completeness": 10,
    "clarity": 9,
    "visual_appeal": 10,
    "insights": 9,
    "final_score": 9,
    "comments": "Well-structured and visually clean. Skill insights are well-targeted. Great use of tables."
}

Make sure your .env contains valid OPENAI and LANGSMITH API keys.
"""

# Load .env vars
dotenv.load_dotenv()

# Check required keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("Missing LANGSMITH_API_KEY in .env")

# Report path
OUTPUT_HTML_PATH = "../outputs/job_market_report_20260113_133015.html"
with open(OUTPUT_HTML_PATH, "r", encoding="utf-8") as f:
    html_content = f.read()

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert evaluator of job market analysis reports."),
    ("user", """
Evaluate the following HTML report on these criteria, and return JSON with keys:
relevance, accuracy, completeness, clarity, visual_appeal, insights, and final_score (out of 10).
Each criterion is scored out of 10 (not 5). Add helpful comments.

Criteria (each scored from 1 to 10):
1. Relevance: Does it match job title + country?
2. Accuracy: Are the mentioned skills actually from job posts?
3. Completeness: Are all required sections included?
4. Clarity: Is the writing clear and structured?
5. Visual Appeal: Is it styled well and pleasant to read?
6. Insights: Are resume suggestions meaningful, actionable, and aligned with job market?

Format:
{{
  "relevance": <int>,         # 1-10
  "accuracy": <int>,          # 1-10
  "completeness": <int>,      # 1-10
  "clarity": <int>,           # 1-10
  "visual_appeal": <int>,     # 1-10
  "insights": <int>,          # 1-10
  "final_score": <int>,       # out of 10
  "comments": "..."
}}

Report HTML:
---------------------
{html}
""")
]).partial(html=html_content)

# LLM
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

# Parser
parser = JsonOutputParser()

# Chain with LangSmith-traceable wrapping
@traceable(name="Evaluate Job Market HTML Report")
def evaluate_html_report():
    chain: Runnable = prompt | llm | parser
    result = chain.invoke({})
    return result

# Entry point
if __name__ == "__main__":
    result = evaluate_html_report()
    print("\n✅ LangSmith Evaluation Summary:\n")
    for k, v in result.items():
        if k == "final_score":
            print(f"⭐ FINAL SCORE (out of 10): {v} ⭐\n")
        else:
            print(f"{k}: {v}")
