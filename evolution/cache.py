"""Pipeline result caching and rate limiting for cost control.

All pipeline runs must go through this module to avoid re-paying for results
that have already been computed. During debugging you will re-run the evolution
loop many times — without caching, each re-run costs the full API price again.

Cache lives in ``.cache/pipeline_runs/`` (gitignored).
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from config import API_CALL_DELAY, CACHE_DIR, ENABLE_CACHE


def get_cache_key(task: str, generation: int) -> str:
    """Return a deterministic cache key for a pipeline run.

    Args:
        task: The user coding request string.
        generation: The tester prompt generation index.

    Returns:
        MD5 hex digest of ``{task}:gen{generation}``.
    """
    return hashlib.md5(f"{task}:gen{generation}".encode()).hexdigest()


def load_cached(task: str, generation: int) -> dict | None:
    """Load a cached pipeline result from disk.

    Args:
        task: The coding task string.
        generation: Tester prompt generation index.

    Returns:
        Cached result dict, or None if not cached or caching is disabled.
    """
    if not ENABLE_CACHE:
        return None
    cache_path = os.path.join(CACHE_DIR, f"{get_cache_key(task, generation)}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return None


def save_to_cache(task: str, generation: int, result: dict) -> None:
    """Write a pipeline result to the disk cache.

    Args:
        task: The coding task string.
        generation: Tester prompt generation index.
        result: Serialisable dict produced by the pipeline run.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{get_cache_key(task, generation)}.json")
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def rate_limited_call(func, *args, **kwargs):
    """Wrap any API call with a delay to avoid hitting Pro plan rate limits.

    Sleeps for ``API_CALL_DELAY`` seconds before invoking ``func``.

    Args:
        func: Callable to invoke after the delay.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Whatever ``func`` returns.
    """
    time.sleep(API_CALL_DELAY)
    return func(*args, **kwargs)
