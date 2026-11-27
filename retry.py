"""Retry with exponential backoff for calls to external APIs (Gemini, Pinecone)."""

from __future__ import annotations

import logging
import random
import re
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
    "ServerError",
    "Timeout",
    "ConnectError",
    "ProtocolError",
)
# Whole-word patterns so request IDs / byte counts in error bodies don't match.
_RETRYABLE_MESSAGE_RE = re.compile(
    r"\b(429|500|502|503|504)\b|rate limit|quota|resource_exhausted|\bunavailable\b"
    r"|overloaded|timed out|deadline exceeded|temporary failure|connection reset",
    re.IGNORECASE,
)
_MAX_CAUSE_DEPTH = 5
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


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """The exception plus its causes: SDK wrappers (e.g. GoogleGenerativeAIError)
    often hide the original status code in ``__cause__``/``__context__``."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < _MAX_CAUSE_DEPTH:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _is_auth_single(exc: BaseException) -> bool:
    if any(code in AUTH_STATUS_CODES for code in _status_codes(exc)):
        return True
    class_name = type(exc).__name__
    if any(m in class_name for m in ("Unauthorized", "PermissionDenied", "Forbidden")):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _AUTH_MESSAGE_MARKERS)


def _is_retryable_single(exc: BaseException) -> bool:
    if any(code in RETRYABLE_STATUS_CODES for code in _status_codes(exc)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    class_name = type(exc).__name__
    if any(marker in class_name for marker in _RETRYABLE_CLASS_MARKERS):
        return True
    return bool(_RETRYABLE_MESSAGE_RE.search(str(exc)))


def is_auth_error(exc: BaseException) -> bool:
    """True if the error (or its cause) looks like an invalid API key or permission problem."""
    return any(_is_auth_single(e) for e in _exception_chain(exc))


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits, transient server/network errors and timeouts (cause chain included)."""
    if isinstance(exc, RAGError):
        return False
    if is_auth_error(exc):
        return False
    return any(_is_retryable_single(e) for e in _exception_chain(exc))


def _short(exc: BaseException, limit: int = 300) -> str:
    """First line of an error message, truncated (full detail goes to the debug log)."""
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


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
            logger.debug("%s raised %s: %s", description, type(exc).__name__, exc)
            if not is_retryable(exc):
                if is_auth_error(exc):
                    raise UpstreamError(
                        f"{description} failed: authentication/permission error "
                        f"({type(exc).__name__}: {_short(exc)}). Check your API key "
                        "(GOOGLE_API_KEY / PINECONE_API_KEY) in .env.",
                        retryable=False,
                    ) from exc
                raise UpstreamError(
                    f"{description} failed: {type(exc).__name__}: {_short(exc)}",
                    retryable=False,
                ) from exc

            if attempt >= max_retries:
                raise UpstreamError(
                    f"{description} failed: Gemini/Pinecone rate limit, overload or timeout "
                    f"after {attempt + 1} attempt(s) ({type(exc).__name__}: {_short(exc)}). "
                    "Wait a minute and retry.",
                    retryable=True,
                ) from exc

            delay = base_delay * (2**attempt) + random.uniform(0, base_delay * 0.1)
            logger.warning(
                "%s failed with retryable error (%s: %s); retry %d/%d in %.1fs",
                description,
                type(exc).__name__,
                _short(exc, 150),
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1
