"""Retry with exponential backoff for calls to external APIs (Gemini, Pinecone)."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

from errors import RAGError, UpstreamError

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
AUTH_STATUS_CODES = frozenset({401, 403})

_RETRYABLE_CLASS_MARKERS = (
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    "TooManyRequests",
    "InternalServerError",
    "Timeout",
)
_RETRYABLE_MESSAGE_MARKERS = (
    "429",
    "rate limit",
    "quota",
    "resource_exhausted",
    "timed out",
    "deadline",
)
_AUTH_MESSAGE_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "permission_denied",
    "unauthorized",
    "invalid api key",
)


def _status_codes(exc: BaseException) -> list[int]:
    """Collect integer HTTP-like status codes exposed by common SDK exceptions."""
    codes: list[int] = []
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if callable(value):  # e.g. grpc's ``code()``
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            codes.append(value)
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and not isinstance(value, bool):
            codes.append(value)
    return codes


def is_auth_error(exc: BaseException) -> bool:
    """True if the error looks like an invalid/missing API key or permission problem."""
    if any(code in AUTH_STATUS_CODES for code in _status_codes(exc)):
        return True
    class_name = type(exc).__name__
    if any(m in class_name for m in ("Unauthorized", "PermissionDenied", "Forbidden")):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _AUTH_MESSAGE_MARKERS)


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits, transient server errors and timeouts."""
    if isinstance(exc, RAGError):
        return False
    if any(code in RETRYABLE_STATUS_CODES for code in _status_codes(exc)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    class_name = type(exc).__name__
    if any(marker in class_name for marker in _RETRYABLE_CLASS_MARKERS):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MESSAGE_MARKERS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    sleep: Callable[[float], None] = time.sleep,
    description: str = "request",
) -> T:
    """Call ``fn()`` retrying transient failures with exponential backoff + jitter.

    * ``RAGError`` subclasses are re-raised untouched.
    * Retryable errors are retried up to ``max_retries`` extra times, waiting
      ``base_delay * 2**attempt + jitter``; then ``UpstreamError(retryable=True)``.
    * Other errors raise ``UpstreamError(retryable=False)`` immediately.
    """
    max_retries = max(0, int(max_retries))
    attempt = 0
    while True:
        try:
            return fn()
        except RAGError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_retryable(exc):
                if is_auth_error(exc):
                    raise UpstreamError(
                        f"{description} failed: authentication/permission error "
                        f"({type(exc).__name__}: {exc}). Check your API key "
                        "(GOOGLE_API_KEY / PINECONE_API_KEY) in .env.",
                        retryable=False,
                    ) from exc
                raise UpstreamError(
                    f"{description} failed: {type(exc).__name__}: {exc}",
                    retryable=False,
                ) from exc

            if attempt >= max_retries:
                raise UpstreamError(
                    f"{description} failed: Gemini/Pinecone rate limit or timeout after "
                    f"{attempt + 1} attempt(s) ({type(exc).__name__}: {exc}). "
                    "Wait a minute and retry.",
                    retryable=True,
                ) from exc

            delay = base_delay * (2**attempt) + random.uniform(0, base_delay * 0.1)
            logger.warning(
                "%s failed with retryable error (%s: %s); retry %d/%d in %.1fs",
                description,
                type(exc).__name__,
                exc,
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1
