"""Tests for retry.call_with_retry / is_retryable / is_auth_error."""

import pytest

from errors import ConfigError, InvalidInputError, UpstreamError
from retry import call_with_retry, is_auth_error, is_retryable


class StatusError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class Flaky:
    """Fails with the given errors in order, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


class SleepRecorder:
    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


@pytest.mark.parametrize(
    "exc",
    [
        StatusError("Too many requests", 429),
        Exception("429 Resource has been exhausted"),
        Exception("You exceeded your current quota"),
        Exception("Rate limit reached"),
        TimeoutError("timed out"),
        StatusError("Service unavailable", 503),
    ],
)
def test_transient_errors_are_retryable(exc):
    assert is_retryable(exc) is True


@pytest.mark.parametrize("exc", [StatusError("Bad request", 400), ValueError("bad request")])
def test_client_errors_are_not_retryable(exc):
    assert is_retryable(exc) is False


def test_auth_errors_are_detected():
    assert is_auth_error(StatusError("Unauthorized", 401)) is True
    assert is_auth_error(ValueError("bad request")) is False


def test_success_on_first_try_does_not_sleep():
    fn, sleep = Flaky(), SleepRecorder()
    assert call_with_retry(fn, max_retries=3, base_delay=1, sleep=sleep) == "ok"
    assert fn.calls == 1 and sleep.delays == []


def test_retries_transient_error_then_succeeds_with_exponential_backoff():
    fn = Flaky(StatusError("429", 429), TimeoutError("slow"))
    sleep = SleepRecorder()
    assert call_with_retry(fn, max_retries=3, base_delay=1, sleep=sleep) == "ok"
    assert fn.calls == 3
    assert len(sleep.delays) == 2
    assert sleep.delays[1] > sleep.delays[0] > 0


def test_exhausted_retries_raise_retryable_upstream_error():
    fn = Flaky(*[StatusError("quota", 429) for _ in range(10)])
    sleep = SleepRecorder()
    with pytest.raises(UpstreamError) as exc_info:
        call_with_retry(fn, max_retries=2, base_delay=0, sleep=sleep, description="Gemini call")
    assert exc_info.value.retryable is True
    assert fn.calls == 3  # first attempt + 2 retries


def test_non_retryable_error_fails_immediately():
    fn = Flaky(StatusError("Bad request", 400))
    sleep = SleepRecorder()
    with pytest.raises(UpstreamError) as exc_info:
        call_with_retry(fn, max_retries=5, base_delay=0, sleep=sleep)
    assert exc_info.value.retryable is False
    assert fn.calls == 1 and sleep.delays == []


def test_value_error_is_wrapped_without_retry():
    fn = Flaky(ValueError("bad request"))
    with pytest.raises(UpstreamError):
        call_with_retry(fn, max_retries=5, base_delay=0, sleep=SleepRecorder())
    assert fn.calls == 1


@pytest.mark.parametrize("err", [ConfigError("cfg"), InvalidInputError("bad q")])
def test_rag_errors_are_reraised_unchanged(err):
    fn = Flaky(err)
    with pytest.raises(type(err)) as exc_info:
        call_with_retry(fn, max_retries=5, base_delay=0, sleep=SleepRecorder())
    assert exc_info.value is err
    assert fn.calls == 1
