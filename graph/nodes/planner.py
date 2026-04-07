"""
graph/nodes/planner.py — Decompose a user message into 1–4 ordered sub-tasks.

ROLE IN THE GRAPH
-----------------
This is the FIRST node on every conversation turn, replacing the old
intent_resolver.  Its output drives all subsequent routing.

The key difference from the old intent_resolver:
  - Old: classified the message into EXACTLY ONE intent.
  - New: decomposes the message into a LIST of 1–4 tasks.

For simple messages the result is still 1 task, and the graph behaves
identically to before.  For compound messages ("analyse X and explain Y")
the graph executes each task in sequence, collects the answers, and
combines them in a single final reply.

ANTI-OVER-DECOMPOSITION
-----------------------
The LLM prompt contains explicit examples and a "if in doubt, use 1 task"
instruction to prevent splitting naturally unified questions.

"What skills do AI and ML engineers need in Germany?" → 1 task (same market)
"Analyse AI engineers in Germany AND Data engineers in France" → 2 tasks
"Explain LangGraph, analyse AI jobs in USA, and review my resume" → 3 tasks

OUTPUTS written to state
------------------------
  task_queue           : list of task dicts [{intent, job_titles, country,
                         focused_topic, query_fragment}, ...]
  accumulated_responses: reset to []  (fresh slate for this turn)
  final_text_response  : reset to None
  html_report_path     : reset to None
  skill_gap_markdown   : reset to None
  market_analysis_markdown : reset to None
  params_confirmed     : reset to False
  report_confirmed     : reset to False
  cache_hit            : reset to False
"""

import logging
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import DEFAULT_TOTAL_POSTS
from graph.state import JobMarketState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a task planner for a Job Market Intelligence System.

Your job is to decompose the user's message into an ordered list of 1–4 tasks.
Each task must use EXACTLY ONE of these four intent types:

  full_market_analysis  — comprehensive market report for given job titles + country
  focused_question      — narrow, specific question answered from market data (cloud platforms, skills, etc.)
  resume_analysis       — compare uploaded resume to market, identify skill gaps
  general_question      — general question answered from AI knowledge (no market data needed)

DECOMPOSITION RULES — read carefully:

1. ONE task per DISTINCT request. If the user asks one thing, create one task.
2. "What skills do AI and ML engineers need in Germany?" → 1 task (same market, not two separate analyses)
3. "Analyse AI jobs in Germany AND Data Engineer jobs in France" → 2 tasks (different markets)
4. "Explain what Docker is" → 1 general_question task — do NOT split this further
5. If a message has both a market data question AND a general knowledge question, those are 2 tasks
6. If in doubt between 1 task and 2, always choose 1
7. Maximum 4 tasks regardless of how many things the user asks

For each task extract:
- intent: one of the four intent types above
- job_titles: list of job roles (e.g. ["AI Engineer", "ML Engineer"]), empty [] for general_question
- country: country name if mentioned, else null
- focused_topic: for focused_question only — the specific topic (e.g. "cloud platforms", "databases")
- total_posts: integer if the user specifies a number of postings (e.g. "use 20 posts", "search 30 jobs"). null if not mentioned.
- query_fragment: the part of the user message this task answers (1 sentence max)

Return valid JSON with a "tasks" key containing the array."""


class Task(BaseModel):
    intent: str = Field(
        description="One of: full_market_analysis, focused_question, resume_analysis, general_question"
    )
    job_titles: list[str] = Field(
        default_factory=list,
        description="Job roles extracted for this task"
    )
    country: Optional[str] = Field(
        default=None,
        description="Country for this task, or null"
    )
    focused_topic: Optional[str] = Field(
        default=None,
        description="Specific topic for focused_question tasks, or null"
    )
    total_posts: Optional[int] = Field(
        default=None,
        description="Number of job postings to collect if mentioned (e.g. 'use 20 posts', 'search 30 jobs'). null if not mentioned."
    )
    query_fragment: str = Field(
        default="",
        description="The part of the user message this task addresses"
    )


class Plan(BaseModel):
    tasks: list[Task] = Field(
        description="Ordered list of 1–4 tasks to execute"
    )


def planner(state: JobMarketState) -> dict:
    """
    Decompose the user's latest message into an ordered task queue.

    Reads the last few messages for context (so corrections like
    "use 10 posts instead" can be understood in context of the prior question).

    Args:
        state: Current graph state.  Reads messages.

    Returns:
        A dict with task_queue, reset output fields, and reset HITL flags.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).with_structured_output(Plan)

    # Pass the last 4 messages (alternating human/AI) so the model has context
    # for corrections like "use 10 posts instead" that refer to a prior question.
    recent_messages = state["messages"][-4:] if len(state["messages"]) >= 4 else state["messages"]

    result: Plan = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        *recent_messages,
    ])

    # Clamp to 1–4 tasks; guard against empty plans
    tasks = result.tasks[:4] if result.tasks else []
    if not tasks:
        # Fallback: treat as a single general question
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "Hello",
        )
        tasks = [Task(intent="general_question", query_fragment=last_human)]

    task_dicts = [t.model_dump() for t in tasks]

    logger.info(
        "planner: decomposed into %d task(s): %s",
        len(task_dicts),
        [(t["intent"], t.get("job_titles"), t.get("country")) for t in task_dicts],
    )

    return {
        "task_queue": task_dicts,
        # Reset per-turn output fields so previous turns don't bleed through
        "accumulated_responses": [],
        "final_text_response": None,
        "html_report_path": None,
        "skill_gap_markdown": None,
        "market_analysis_markdown": None,
        # Reset HITL flags
        "params_confirmed": False,
        "report_confirmed": False,
        "cache_hit": False,
    }
