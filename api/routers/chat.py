"""
api/routers/chat.py — Chat endpoints with SSE streaming.

Endpoints:
  POST /api/chat               — Start a new conversation turn.
  POST /api/chat/{session_id}/reply — Resume after a HITL interrupt.

SSE event types emitted to the client:
  node_start  {"node": "<name>"}                          — graph node began
  token       {"content": "<text>"}                       — LLM streaming token
  interrupt   {"prompt": "<text>", "session_id": "<id>"}  — graph paused at HITL
  done        {"session_id", "final_text_response",        — graph completed
               "html_report_path", "intent", "cache_hit"}
  error       {"detail": "<message>"}                     — unhandled exception

HITL flow:
  1. Client POSTs to /api/chat → SSE stream begins.
  2. When graph hits interrupt(), astream_events() exhausts naturally.
  3. Server calls get_state() → detects snapshot.next is truthy → emits `interrupt`.
  4. Client shows the prompt, captures user reply, POSTs to /api/chat/{id}/reply.
  5. Server resumes graph with Command(resume=reply) → new SSE stream.
  6. Repeat until graph emits `done`.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.dependencies import get_graph

logger = logging.getLogger(__name__)

router = APIRouter()

# Names registered via add_node() in graph/graph.py.
# Used to filter astream_events so we only emit node_start for actual graph nodes,
# not internal LangChain chains.
GRAPH_NODE_NAMES = {
    "intent_resolver",
    "check_resume",
    "resume_parser",
    "cache_lookup",
    "confirm_search_params",
    "job_collector",
    "requirements_extractor",
    "market_analyzer",
    "skill_gap_analyzer",
    "answer_focused",
    "confirm_report_format",
    "html_report_generator",
    "respond",
}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # generated if absent
    total_posts: int = 30


class ReplyRequest(BaseModel):
    reply: str


# Only stream LLM tokens from nodes where the output is the final user-facing
# text. Structured-output nodes (intent_resolver, requirements_extractor) emit
# raw JSON tokens that are meaningless to the client.
_TOKEN_NODES = {
    "respond",
    "market_analyzer",
    "skill_gap_analyzer",
    "answer_focused",
    "html_report_generator",
}


async def stream_graph_run(graph, payload, config: dict):
    """
    Core SSE generator for both /chat and /chat/{id}/reply.

    Iterates graph.astream_events(), emitting node_start and token events.
    After the stream exhausts, checks aget_state() to detect a HITL interrupt
    and emits either `interrupt` or `done` as the final event.
    """
    try:
        # Deduplicate node_start by (node_name, langgraph_step): LangGraph fires
        # on_chain_start at multiple chain levels within each node execution, all
        # sharing the same langgraph_node + langgraph_step metadata values.
        seen_steps: set = set()

        async for event in graph.astream_events(payload, config, version="v2"):
            kind = event["event"]
            meta = event.get("metadata", {})
            node_name = meta.get("langgraph_node")

            if kind == "on_chain_start" and node_name in GRAPH_NODE_NAMES:
                step_key = (node_name, meta.get("langgraph_step"))
                if step_key not in seen_steps:
                    seen_steps.add(step_key)
                    logger.info("node_start: %s", node_name)
                    yield {
                        "event": "node_start",
                        "data": json.dumps({"node": node_name}),
                    }

            elif kind == "on_chat_model_stream" and node_name in _TOKEN_NODES:
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    yield {
                        "event": "token",
                        "data": json.dumps({"content": chunk.content}),
                    }

        # Stream exhausted — check whether the graph paused at a HITL interrupt
        # or completed normally.
        logger.info("stream: astream_events loop exhausted, calling aget_state")
        snapshot = await asyncio.wait_for(graph.aget_state(config), timeout=15)
        logger.info("stream: aget_state done, next=%s", bool(snapshot.next))

        if snapshot.next:
            # Graph is paused. Extract the interrupt prompt from the first task.
            prompt = snapshot.tasks[0].interrupts[0].value
            yield {
                "event": "interrupt",
                "data": json.dumps({
                    "prompt": prompt,
                    "session_id": config["configurable"]["thread_id"],
                }),
            }
        else:
            state = snapshot.values
            yield {
                "event": "done",
                "data": json.dumps({
                    "session_id": config["configurable"]["thread_id"],
                    "final_text_response": state.get("final_text_response"),
                    "html_report_path": state.get("html_report_path"),
                    "intent": state.get("intent"),
                    "cache_hit": state.get("cache_hit", False),
                }),
            }

    except Exception as e:
        logger.exception("stream: unhandled exception: %s", e)
        yield {
            "event": "error",
            "data": json.dumps({"detail": str(e)}),
        }


@router.post("/chat")
async def chat(req: ChatRequest, graph=Depends(get_graph)):
    """
    Start a new conversation turn.

    Generates a session_id if one is not provided. Returns an SSE stream
    that emits node_start, token, and finally interrupt or done.
    """
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    # If this session is stuck at a HITL interrupt (e.g. the user left mid-flow
    # and returned later), start a fresh session rather than blocking the user.
    snapshot = await graph.aget_state(config)
    if snapshot.next:
        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "session_id": session_id,
        "total_posts": req.total_posts,
        "params_confirmed": False,
        "report_confirmed": False,
        "cache_hit": False,
    }

    logger.info("chat: starting SSE stream for session %s", session_id[:8])
    return EventSourceResponse(
        stream_graph_run(graph, initial_state, config),
        ping=15,  # keep-alive every 15s; prevents proxy timeouts during long LLM/SerpAPI calls
        headers={"X-Accel-Buffering": "no"},  # tell proxies (nginx/envoy) not to buffer SSE
    )


@router.post("/chat/{session_id}/reply")
async def reply(session_id: str, req: ReplyRequest, graph=Depends(get_graph)):
    """
    Resume a graph that is paused at a HITL interrupt.

    Returns a new SSE stream that continues from where the graph left off.
    """
    config = {"configurable": {"thread_id": session_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' is not paused at an interrupt.",
        )

    logger.info("reply: resuming session %s", session_id[:8])
    return EventSourceResponse(
        stream_graph_run(graph, Command(resume=req.reply), config),
        ping=15,
        headers={"X-Accel-Buffering": "no"},
    )
