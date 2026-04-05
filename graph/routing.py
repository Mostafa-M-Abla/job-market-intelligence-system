"""
graph/routing.py — Conditional edge routing functions for the StateGraph.

In LangGraph, "edges" are the connections between nodes (steps).  Most edges
are fixed — node A always leads to node B.  But "conditional edges" let the
graph choose the next node dynamically based on the current state.

Each function in this file:
  1. Receives the current state after a node has finished running.
  2. Inspects a few fields (intent, cache_hit, flags, etc.).
  3. Returns a string — the name of the node that should run next.

These functions contain NO business logic and make NO LLM calls.
They are pure routing decisions, which is why they are easy to unit-test
(see tests/test_graph_routing.py).
"""

from graph.session_store import has_resume
from graph.state import JobMarketState


# ── After intent_resolver ──────────────────────────────────────────────────────

def route_after_intent(state: JobMarketState) -> str:
    """
    First routing decision — determines which major path to take based on
    what the user is trying to do.

    Routes:
      general_question     -> respond        (LLM answers directly, no tools needed)
      resume_analysis      -> check_resume   (need to verify a PDF was uploaded first)
      full_market_analysis -> cache_lookup   (check DB before hitting SerpAPI)
      focused_question     -> cache_lookup   (same — check DB first)
    """
    intent = state.get("intent")

    if intent == "general_question":
        # No market data needed — go straight to the final response node.
        return "respond"

    if intent == "resume_analysis":
        # Before doing anything else, verify the user has uploaded a resume.
        return "check_resume"

    if intent in ("full_market_analysis", "focused_question"):
        # Both of these need market data. Check the cache first to avoid
        # unnecessary SerpAPI calls.
        return "cache_lookup"

    # Unknown or None intent — respond with a fallback message.
    return "respond"


# ── After check_resume ────────────────────────────────────────────────────────

def route_after_check_resume(state: JobMarketState) -> str:
    """
    Decides whether to proceed with resume analysis or ask the user to upload
    their resume first.

    This is a deterministic check — no LLM involved, just a dictionary lookup.

    Routes:
      resume found   -> resume_parser  (extract text from the PDF)
      resume missing -> respond        (the respond node will surface the upload prompt
                                        that check_resume wrote to final_text_response)
    """
    session_id = state.get("session_id", "")
    if has_resume(session_id):
        return "resume_parser"
    # No resume on file — the check_resume node already wrote a helpful message
    # into final_text_response, so respond will display it to the user.
    return "respond"


# ── After cache_lookup ────────────────────────────────────────────────────────

def route_after_cache_lookup(state: JobMarketState) -> str:
    """
    After checking the cache, decides whether to use the stored data or
    kick off a fresh job search.

    On a CACHE HIT  — skip the expensive SerpAPI + LLM pipeline entirely.
                      Route directly to wherever the intent needs to go next.
    On a CACHE MISS — ask the user to confirm the search parameters before
                      spending SerpAPI credits (HITL step).

    Routes (cache hit):
      focused_question     -> answer_focused      (answer from cached market data)
      resume_analysis      -> skill_gap_analyzer  (compare resume to cached market data)
      full_market_analysis -> confirm_report_format (ask: HTML or text?)

    Routes (cache miss):
      any intent           -> confirm_search_params (HITL: show params, wait for OK)
    """
    intent = state.get("intent")
    cache_hit = state.get("cache_hit", False)

    if cache_hit:
        # Market data already in DB — no SerpAPI call needed.
        # Also skip confirm_search_params since there is nothing to confirm.
        if intent == "focused_question":
            return "answer_focused"
        if intent == "resume_analysis":
            return "skill_gap_analyzer"
        # full_market_analysis — go ask the user what output format they want
        return "confirm_report_format"

    # No cached data — we need to search. But first ask the user to confirm
    # the parameters (job titles, country, post count) to avoid surprise costs.
    return "confirm_search_params"


# ── After confirm_search_params (HITL) ───────────────────────────────────────

def route_after_confirm_search_params(state: JobMarketState) -> str:
    """
    After the HITL pause in confirm_search_params, decides what to do next.

    If the user confirmed  -> start the job collection pipeline.
    If the user edited     -> go back to intent_resolver so the LLM can
                              re-parse the corrected message and update
                              job_titles / country accordingly.

    Routes:
      params_confirmed = True  -> job_collector    (begin SerpAPI search)
      params_confirmed = False -> intent_resolver  (re-classify with updated params)
    """
    if state.get("params_confirmed"):
        return "job_collector"
    # User changed something — feed their new message back through intent_resolver
    # so it can extract the corrected job titles / country.
    return "intent_resolver"


# ── After market_analyzer ─────────────────────────────────────────────────────

def route_after_market_analyzer(state: JobMarketState) -> str:
    """
    After the market analysis is complete, decides what to do with the results
    based on the original intent.

    Routes:
      focused_question     -> answer_focused      (extract the specific answer the user asked for)
      resume_analysis      -> skill_gap_analyzer  (compare resume against the fresh market data)
      full_market_analysis -> confirm_report_format (ask: HTML or text?)
    """
    intent = state.get("intent")

    if intent == "focused_question":
        return "answer_focused"
    if intent == "resume_analysis":
        return "skill_gap_analyzer"
    # full_market_analysis
    return "confirm_report_format"


# ── After skill_gap_analyzer ─────────────────────────────────────────────────

def route_after_skill_gap(state: JobMarketState) -> str:
    """
    After the skill gap analysis is ready, always ask the user whether they
    want a full HTML report or a text summary.

    Routes:
      always -> confirm_report_format
    """
    return "confirm_report_format"


# ── After confirm_report_format (HITL) ───────────────────────────────────────

def route_after_confirm_report_format(state: JobMarketState) -> str:
    """
    After the user chooses their preferred output format, routes to the
    appropriate generator.

    Routes:
      report_confirmed = True  -> html_report_generator  (build a styled HTML file)
      report_confirmed = False -> answer_focused          (write a concise text reply)
    """
    if state.get("report_confirmed"):
        return "html_report_generator"
    # User prefers a text summary — answer_focused will compose a concise reply.
    return "answer_focused"


# ── After resume_parser ───────────────────────────────────────────────────────

def route_after_resume_parser(state: JobMarketState) -> str:
    """
    After attempting to extract text from the resume PDF, decides whether
    extraction succeeded or failed.

    If final_text_response is already set it means resume_parser encountered
    an error and wrote an error message — surface that to the user immediately.

    Routes:
      success -> cache_lookup  (now check if we have market data before searching)
      failure -> respond       (display the error message)
    """
    if state.get("final_text_response"):
        # resume_parser set an error message — bail out gracefully.
        return "respond"
    return "cache_lookup"
