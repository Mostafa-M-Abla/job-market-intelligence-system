"""
Non-interactive Phase 1 integration test runner.
Drives all 4 intents, handling HITL interrupts programmatically.

Usage:
    python run_tests.py              # runs all tests
    python run_tests.py general      # only general_question
    python run_tests.py focused      # only focused_question
    python run_tests.py market       # only full_market_analysis
    python run_tests.py resume       # only resume_analysis (needs Resume.pdf)
"""

import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from db.connection import init_db
from graph.graph import build_graph
from graph.session_store import save_resume

# ── Setup ──────────────────────────────────────────────────────────────────────
init_db()
graph = build_graph()

SEPARATOR = "=" * 65


def _print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def _get_interrupt_value(snapshot) -> str | None:
    """Extract the interrupt prompt from the graph snapshot if paused."""
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            return task.interrupts[0].value
    return None


def run_conversation(session_id: str, message: str, hitl_replies: list[str]) -> dict:
    """
    Run a full conversation turn, automatically supplying hitl_replies
    in order whenever an interrupt is encountered.
    """
    config = {"configurable": {"thread_id": session_id}}
    replies_iter = iter(hitl_replies)

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "total_posts": 5,  # Low for testing — saves API credits
        "params_confirmed": False,
        "report_confirmed": False,
        "cache_hit": False,
    }

    result = graph.invoke(initial_state, config)

    # Handle any HITL interrupts
    for _ in range(5):  # max 5 interrupt rounds
        snapshot = graph.get_state(config)
        interrupt_prompt = _get_interrupt_value(snapshot)

        if not interrupt_prompt:
            break

        # Get the next programmed reply
        try:
            reply = next(replies_iter)
        except StopIteration:
            print(f"  [WARN] Unexpected interrupt with no reply prepared:")
            print(f"         {interrupt_prompt}")
            break

        print(f"\n  [INTERRUPT] {interrupt_prompt}")
        print(f"  [AUTO-REPLY] {reply}")

        result = graph.invoke(Command(resume=reply), config)

    return result


def print_result(result: dict):
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = last.content
        # Truncate very long responses for readability
        if len(content) > 800:
            print(f"\n  [Assistant] {content[:800]}\n  ... (truncated, {len(content)} chars total)")
        else:
            print(f"\n  [Assistant] {content}")
    if result.get("html_report_path"):
        print(f"\n  [Report] {result['html_report_path']}")
    if result.get("cache_hit"):
        print(f"\n  [Cache] HIT — market data loaded from cache")


# ── Test 1: General Question ───────────────────────────────────────────────────

def test_general_question():
    _print_section("TEST 1: General Question (LLM only, no tools)")
    session_id = f"test-general-{uuid.uuid4().hex[:6]}"
    print(f"  Session: {session_id}")
    print(f"  Query  : What is the difference between LangGraph and CrewAI?")

    result = run_conversation(
        session_id=session_id,
        message="What is the difference between LangGraph and CrewAI?",
        hitl_replies=[],  # No HITL expected
    )
    print_result(result)
    print(f"\n  [OK] general_question test complete")


# ── Test 2: Focused Question ───────────────────────────────────────────────────

def test_focused_question():
    _print_section("TEST 2: Focused Question (market data + HITL)")
    session_id = f"test-focused-{uuid.uuid4().hex[:6]}"
    print(f"  Session: {session_id}")
    print(f"  Query  : What are the most in-demand cloud platforms for AI Engineer jobs in Germany?")

    result = run_conversation(
        session_id=session_id,
        message="What are the most in-demand cloud platforms for AI Engineer jobs in Germany?",
        hitl_replies=["confirm"],  # Reply to confirm_search_params
    )
    print_result(result)
    print(f"\n  [OK] focused_question test complete")


# ── Test 3: Full Market Analysis ──────────────────────────────────────────────

def test_full_market_analysis():
    _print_section("TEST 3: Full Market Analysis (full pipeline + 2x HITL)")
    session_id = f"test-market-{uuid.uuid4().hex[:6]}"
    print(f"  Session: {session_id}")
    print(f"  Query  : Do a full market analysis for AI Engineer roles in Germany")

    result = run_conversation(
        session_id=session_id,
        message="Do a full market analysis for AI Engineer roles in Germany",
        hitl_replies=[
            "confirm",  # confirm_search_params
            "B",        # confirm_report_format -> text summary (faster, no HTML gen cost)
        ],
    )
    print_result(result)
    print(f"\n  [OK] full_market_analysis test complete")


# ── Test 4: Full Market Analysis with HTML Report ─────────────────────────────

def test_full_market_analysis_with_report():
    _print_section("TEST 4: Full Market Analysis -> HTML Report")
    session_id = f"test-report-{uuid.uuid4().hex[:6]}"
    print(f"  Session: {session_id}")
    print(f"  Query  : Analyse the ML Engineer job market in the UK")

    result = run_conversation(
        session_id=session_id,
        message="Analyse the ML Engineer job market in the UK",
        hitl_replies=[
            "confirm",  # confirm_search_params
            "A",        # confirm_report_format -> HTML report
        ],
    )
    print_result(result)
    print(f"\n  [OK] full_market_analysis + HTML report test complete")


# ── Test 5: Cache Hit ─────────────────────────────────────────────────────────

def test_cache_hit(prior_session_id: str):
    """Re-uses the market data from test_full_market_analysis (same titles + country)."""
    _print_section("TEST 5: Cache Hit (second request, same params)")
    session_id = f"test-cache-{uuid.uuid4().hex[:6]}"
    print(f"  Session: {session_id}")
    print(f"  Query  : What frameworks are most popular for AI engineers in Germany?")
    print(f"  (Expecting cache HIT — no SerpAPI call)")

    result = run_conversation(
        session_id=session_id,
        message="What frameworks are most popular for AI engineers in Germany?",
        hitl_replies=[
            # No confirm_search_params interrupt expected on cache hit
            # No report format interrupt for focused_question
        ],
    )
    print_result(result)
    print(f"\n  [OK] cache hit test complete")


# ── Test 6: Resume Analysis ───────────────────────────────────────────────────

def test_resume_analysis(resume_path: str = "Resume.pdf"):
    _print_section("TEST 6: Resume Analysis (resume + market data + HITL)")
    session_id = f"test-resume-{uuid.uuid4().hex[:6]}"
    print(f"  Session: {session_id}")

    path = Path(resume_path)
    if not path.exists():
        print(f"  [SKIP] Resume not found at {resume_path}. Pass path as argument.")
        return None

    save_resume(session_id, path.read_bytes())
    print(f"  Resume : {resume_path} loaded ({path.stat().st_size} bytes)")
    print(f"  Query  : What skills should I learn next to get an AI Engineer job in Germany?")

    result = run_conversation(
        session_id=session_id,
        message="What skills should I learn next to get an AI Engineer job in Germany?",
        hitl_replies=[
            "confirm",  # confirm_search_params (if cache miss)
            "B",        # confirm_report_format -> text summary
        ],
    )
    print_result(result)
    print(f"\n  [OK] resume_analysis test complete")
    return session_id


# ── Test 7: Resume Missing ────────────────────────────────────────────────────

def test_resume_missing():
    _print_section("TEST 7: Resume Analysis — Resume Not Uploaded")
    session_id = f"test-noresume-{uuid.uuid4().hex[:6]}"
    print(f"  Session: {session_id}")
    print(f"  Query  : Compare my resume to the AI Engineer market in Germany")
    print(f"  (No resume uploaded — expect upload prompt)")

    result = run_conversation(
        session_id=session_id,
        message="Compare my resume to the AI Engineer market in Germany",
        hitl_replies=[],  # No HITL — should immediately ask for resume
    )
    print_result(result)
    print(f"\n  [OK] resume_missing test complete")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    resume_path = sys.argv[2] if len(sys.argv) > 2 else "Resume.pdf"

    print(f"\nPhase 1 Integration Tests")
    print(f"{'='*65}")
    print(f"Arg: {arg}  |  Resume: {resume_path}")

    if arg in ("general", "all"):
        test_general_question()

    if arg in ("focused", "all"):
        test_focused_question()

    if arg in ("market", "all"):
        test_full_market_analysis()

    if arg in ("report", "all"):
        test_full_market_analysis_with_report()

    if arg in ("cache", "all"):
        # Cache test works best after focused/market test has run
        test_cache_hit(prior_session_id="")

    if arg in ("resume_missing", "all"):
        test_resume_missing()

    if arg in ("resume", "all"):
        test_resume_analysis(resume_path)

    print(f"\n{'='*65}")
    print("  All selected tests finished.")
    print(f"{'='*65}")
