"""
Pure routing unit tests — no LLM calls, no DB, no SerpAPI.
Tests that routing functions return the correct next node for all state combinations.
"""

import pytest

from graph.routing import (
    route_after_cache_lookup,
    route_after_check_resume,
    route_after_confirm_report_format,
    route_after_confirm_search_params,
    route_after_intent,
    route_after_job_collector,
    route_after_market_analyzer,
    route_after_resume_parser,
    route_after_skill_gap,
)


def _state(**kwargs):
    """Build a minimal JobMarketState dict for testing."""
    defaults = {
        "messages": [],
        "session_id": "test-session",
        "intent": None,
        "focused_topic": None,
        "job_titles": ["AI Engineer"],
        "country": "Germany",
        "total_posts": 30,
        "params_confirmed": False,
        "report_confirmed": False,
        "raw_job_postings": None,
        "task_queue": [],
        "accumulated_responses": [],
        "extracted_requirements": None,
        "market_analysis_markdown": None,
        "resume_text": None,
        "resume_skills": None,
        "skill_gap_markdown": None,
        "html_report_path": None,
        "final_text_response": None,
        "cache_hit": False,
        "cache_key": None,
    }
    defaults.update(kwargs)
    return defaults


# ── route_after_intent ────────────────────────────────────────────────────────

class TestRouteAfterIntent:
    def test_general_question_goes_to_answer_general(self):
        assert route_after_intent(_state(intent="general_question")) == "answer_general"

    def test_resume_analysis_goes_to_check_resume(self):
        assert route_after_intent(_state(intent="resume_analysis")) == "check_resume"

    def test_full_market_analysis_goes_to_cache_lookup(self):
        assert route_after_intent(_state(intent="full_market_analysis")) == "cache_lookup"

    def test_focused_question_goes_to_cache_lookup(self):
        assert route_after_intent(_state(intent="focused_question")) == "cache_lookup"

    def test_unknown_intent_goes_to_answer_general(self):
        assert route_after_intent(_state(intent=None)) == "answer_general"


# ── route_after_check_resume ──────────────────────────────────────────────────

class TestRouteAfterCheckResume:
    def test_no_resume_goes_to_respond(self, monkeypatch):
        monkeypatch.setattr("graph.routing.has_resume", lambda sid: False)
        assert route_after_check_resume(_state()) == "respond"

    def test_resume_exists_goes_to_resume_parser(self, monkeypatch):
        monkeypatch.setattr("graph.routing.has_resume", lambda sid: True)
        assert route_after_check_resume(_state()) == "resume_parser"


# ── route_after_cache_lookup ──────────────────────────────────────────────────

class TestRouteAfterCacheLookup:
    def test_cache_miss_goes_to_confirm_search_params(self):
        assert route_after_cache_lookup(_state(cache_hit=False)) == "confirm_search_params"

    def test_cache_hit_focused_question_goes_to_answer_focused(self):
        result = route_after_cache_lookup(_state(cache_hit=True, intent="focused_question"))
        assert result == "answer_focused"

    def test_cache_hit_resume_analysis_goes_to_skill_gap_analyzer(self):
        result = route_after_cache_lookup(_state(cache_hit=True, intent="resume_analysis"))
        assert result == "skill_gap_analyzer"

    def test_cache_hit_full_analysis_goes_to_confirm_report_format(self):
        result = route_after_cache_lookup(_state(cache_hit=True, intent="full_market_analysis"))
        assert result == "confirm_report_format"


# ── route_after_confirm_search_params ────────────────────────────────────────

class TestRouteAfterConfirmSearchParams:
    def test_confirmed_goes_to_job_collector(self):
        assert route_after_confirm_search_params(_state(params_confirmed=True)) == "job_collector"

    def test_not_confirmed_goes_to_planner(self):
        assert route_after_confirm_search_params(_state(params_confirmed=False)) == "planner"


# ── route_after_job_collector ────────────────────────────────────────────────

class TestRouteAfterJobCollector:
    def test_postings_found_goes_to_requirements_extractor(self):
        state = _state(raw_job_postings=[{"title": "AI Engineer"}])
        assert route_after_job_collector(state) == "requirements_extractor"

    def test_no_postings_last_task_goes_to_respond(self):
        state = _state(raw_job_postings=[], task_queue=[])
        assert route_after_job_collector(state) == "respond"

    def test_no_postings_more_tasks_goes_to_task_dispatcher(self):
        state = _state(raw_job_postings=[], task_queue=[{"intent": "general_question"}])
        assert route_after_job_collector(state) == "task_dispatcher"

    def test_none_postings_treated_as_empty(self):
        state = _state(raw_job_postings=None, task_queue=[])
        assert route_after_job_collector(state) == "respond"


# ── route_after_market_analyzer ──────────────────────────────────────────────

class TestRouteAfterMarketAnalyzer:
    def test_focused_question_goes_to_answer_focused(self):
        result = route_after_market_analyzer(_state(intent="focused_question"))
        assert result == "answer_focused"

    def test_resume_analysis_goes_to_skill_gap_analyzer(self):
        result = route_after_market_analyzer(_state(intent="resume_analysis"))
        assert result == "skill_gap_analyzer"

    def test_full_market_analysis_goes_to_confirm_report_format(self):
        result = route_after_market_analyzer(_state(intent="full_market_analysis"))
        assert result == "confirm_report_format"


# ── route_after_skill_gap ─────────────────────────────────────────────────────

class TestRouteAfterSkillGap:
    def test_always_goes_to_confirm_report_format(self):
        assert route_after_skill_gap(_state()) == "confirm_report_format"


# ── route_after_confirm_report_format ────────────────────────────────────────

class TestRouteAfterConfirmReportFormat:
    def test_report_confirmed_goes_to_html_generator(self):
        result = route_after_confirm_report_format(_state(report_confirmed=True))
        assert result == "html_report_generator"

    def test_report_declined_goes_to_answer_focused(self):
        result = route_after_confirm_report_format(_state(report_confirmed=False))
        assert result == "answer_focused"


# ── route_after_resume_parser ─────────────────────────────────────────────────

class TestRouteAfterResumeParser:
    def test_no_error_goes_to_cache_lookup(self):
        result = route_after_resume_parser(_state(final_text_response=None))
        assert result == "cache_lookup"

    def test_parse_error_goes_to_respond(self):
        result = route_after_resume_parser(_state(final_text_response="Error: no text found"))
        assert result == "respond"
