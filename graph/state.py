"""
graph/state.py — Shared state definition for the entire LangGraph graph.

In LangGraph every node (step) in the graph reads from and writes to a single
shared state object.  Think of it like a whiteboard that all agents can read
and update — each node only writes back the fields it changes, and LangGraph
merges those changes in automatically.

`JobMarketState` is a TypedDict, meaning it is just a Python dict with type
annotations.  LangGraph serialises this state to the database after every node
so that long-running conversations can survive server restarts and the graph
can be paused (for human-in-the-loop) and resumed later.
"""

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class JobMarketState(TypedDict):
    """
    The single source of truth shared across all graph nodes.

    Every node receives the full state and returns a dict containing only the
    fields it wants to update.  LangGraph merges those updates back into this
    state automatically before passing it to the next node.

    Sections
    --------
    Conversation  — the chat history and which session this belongs to.
    Intent        — what the user is trying to do and any extracted parameters.
    Search params — the job titles / country the user wants to analyse.
    HITL flags    — booleans that track whether the user has confirmed actions.
    Data payloads — intermediate results produced during the pipeline.
    Output        — the final answer / report that will be shown to the user.
    Cache         — metadata about whether we reused a previous analysis.
    """

    # ── Conversation ─────────────────────────────────────────────────────────

    # Full chat history between the user and the assistant.
    # The special `add_messages` reducer means LangGraph *appends* new messages
    # to this list rather than replacing it — so history is never lost.
    messages: Annotated[list[BaseMessage], add_messages]

    # A unique identifier for this user's session (UUID).
    # Used to keep different users' data completely separate.
    session_id: str

    # ── Intent resolution ─────────────────────────────────────────────────────

    # What the user wants to do, determined by the intent_resolver node.
    # One of four values:
    #   "full_market_analysis" — run a complete market analysis + optional HTML report
    #   "focused_question"     — answer a single narrow question from market data
    #   "resume_analysis"      — compare the user's resume against the market
    #   "general_question"     — answer a general question from the LLM's own knowledge
    intent: Optional[Literal[
        "full_market_analysis",
        "focused_question",
        "resume_analysis",
        "general_question",
    ]]

    # For focused_question only: the specific topic the user is asking about.
    # e.g. "cloud platforms", "databases", "front-end frameworks"
    # This tells answer_focused which part of the market data to zoom in on.
    focused_topic: Optional[str]

    # ── Search parameters (confirmed via HITL before any SerpAPI call) ────────

    # The job roles to search for, e.g. ["AI Engineer", "ML Engineer"].
    # Extracted from the user's message by intent_resolver.
    job_titles: Optional[list[str]]

    # The country to search in, e.g. "Germany".
    country: Optional[str]

    # How many job postings to collect from SerpAPI.
    # Defaults to 30 (configurable via DEFAULT_TOTAL_POSTS env var).
    total_posts: int

    # ── HITL flags ────────────────────────────────────────────────────────────
    # HITL = Human In The Loop.  Before expensive or irreversible actions the
    # graph pauses and asks the user for confirmation.  These booleans track
    # whether the user has given that confirmation.

    # Set to True once the user confirms (or edits) the search parameters
    # in the confirm_search_params node.  Only then does the job collector run.
    params_confirmed: bool

    # Set to True if the user chose "A" (HTML report) at the confirm_report_format
    # node, or False if they chose "B" (text summary in chat).
    report_confirmed: bool

    # ── Data payloads ─────────────────────────────────────────────────────────
    # These fields are populated step-by-step as the pipeline executes.

    # Raw job postings fetched from SerpAPI — each dict contains title, company,
    # location, description, job_id, etc.
    raw_job_postings: Optional[list[dict]]

    # Structured requirements extracted from each job posting by the LLM.
    # Each dict contains: job_id, title, company, technical_skills,
    # cloud_platforms, certifications.
    extracted_requirements: Optional[list[dict]]

    # The aggregated market analysis as a Markdown string, produced by
    # market_analyzer.  Contains skill frequency tables, cloud platform
    # rankings, and key insights.
    market_analysis_markdown: Optional[str]

    # Plain text extracted from the user's uploaded resume PDF.
    resume_text: Optional[str]

    # Structured skills extracted from the resume (populated by skill_gap_analyzer).
    resume_skills: Optional[dict]

    # The personalised skill gap analysis as a Markdown string — compares what
    # the user already knows against what the market demands.
    skill_gap_markdown: Optional[str]

    # ── Output ────────────────────────────────────────────────────────────────

    # URL path to the saved HTML report file, e.g.
    # "/api/reports/session-abc/job_market_report_20260405.html".
    # Only populated when the user chose option A (HTML report).
    html_report_path: Optional[str]

    # A plain-text or Markdown response to show directly in the chat.
    # Used for focused answers, text summaries, and error/prompt messages.
    final_text_response: Optional[str]

    # ── Cache metadata ────────────────────────────────────────────────────────

    # True if market analysis data was loaded from the DB cache rather than
    # fetched fresh from SerpAPI.  Used by routing to skip confirmation steps.
    cache_hit: bool

    # The SHA-256 key used to look up / store this analysis in the cache.
    # Built from sorted(job_titles) + country + today's date.
    cache_key: Optional[str]

    # ── Multi-task planner ────────────────────────────────────────────────────

    # Queue of remaining sub-tasks produced by the planner node.
    # Each entry is a dict: {intent, job_titles, country, focused_topic, query_fragment}.
    # task_dispatcher pops the first entry and sets the routing state fields.
    # An empty list means all tasks for this turn have been completed.
    task_queue: list[dict]

    # Partial answers collected as each sub-task completes.
    # answer_general, answer_focused, and html_report_generator each append one
    # entry here.  The respond node combines them into a single final reply.
    accumulated_responses: list[str]
