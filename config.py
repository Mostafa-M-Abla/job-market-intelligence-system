"""
config.py — Application configuration constants.

Values that control business logic belong here, not in .env.
.env is reserved for secrets (API keys) and environment-specific paths (DB_PATH, OUTPUTS_DIR).
"""

# Number of days a cached market analysis stays valid before it must be refreshed.
CACHE_TTL_DAYS: int = 7

# Default number of job postings to collect per SerpAPI run when the caller
# does not specify a value. Kept low in tests (total_posts=5) to save credits.
DEFAULT_TOTAL_POSTS: int = 30