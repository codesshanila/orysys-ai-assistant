import pytest

from app.errors import RateLimitExceeded
from app.security.rate_limit import TokenBucketLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


async def test_burst_then_refill():
    clock = FakeClock()
    limiter = TokenBucketLimiter(capacity=3, refill_per_minute=60, clock=clock)  # 1 token per second
    for _ in range(3):
        await limiter.acquire("u")
    with pytest.raises(RateLimitExceeded) as exc:
        await limiter.acquire("u")
    assert exc.value.retry_after_s == pytest.approx(1.0)
    clock.now = 1.0
    await limiter.acquire("u")


async def test_users_are_isolated():
    limiter = TokenBucketLimiter(capacity=1, refill_per_minute=1, clock=FakeClock())
    await limiter.acquire("alice")
    await limiter.acquire("bob")
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire("alice")


def test_invalid_configuration():
    with pytest.raises(ValueError):
        TokenBucketLimiter(capacity=0, refill_per_minute=1)
