"""
api/main.py — FastAPI application entry point.

Responsibilities:
  - Build the FastAPI app with lifespan (graph singleton init).
  - Configure CORS (locked to specific origins via CORS_ORIGINS env var).
  - Mount static file serving for generated HTML reports.
  - Register all routers under the /api prefix.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.dependencies import startup, shutdown
from api.routers import chat, resume, history


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(
    title="Job Market Intelligence API",
    description="Agentic job market analysis via LangGraph + SSE streaming.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — defaults to * for local dev; set CORS_ORIGINS in production.
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for generated HTML reports.
# html_report_saver.py writes to OUTPUTS_DIR/{session_id}/filename.html
# and returns "/api/reports/{session_id}/{filename}" as the URL.
outputs_dir = Path(os.getenv("OUTPUTS_DIR", "outputs"))
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/reports", StaticFiles(directory=str(outputs_dir)), name="reports")

app.include_router(chat.router, prefix="/api")
app.include_router(resume.router, prefix="/api")
app.include_router(history.router, prefix="/api")
