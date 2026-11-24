"""Per-user token bucket rate limiter.

Each user has a bucket of `capacity` tokens that refills continuously at
`refill_per_minute`. A request costs 1 token (deep research costs more - it fans out to
many LLM calls). Empty bucket -> RateLimitExceeded with an exact Retry-After.

In-memory and asyncio-safe, which is right for a single-instance POC. Multi-instance
deployments would keep the same algorithm in Redis (one Lua script) - see decisions.md.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.errors import RateLimitExceeded


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_per_minute: float, clock=time.monotonic):
        if capacity <= 0 or refill_per_minute <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self.capacity = float(capacity)
        self.rate_per_s = refill_per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    def _refill(self, bucket: _Bucket, now: float) -> None:
        bucket.tokens = min(self.capacity, bucket.tokens + (now - bucket.updated) * self.rate_per_s)
        bucket.updated = now

    async def acquire(self, user: str, cost: float = 1.0) -> float:
        """Consume `cost` tokens or raise. Returns tokens remaining."""
        async with self._lock:
            now = self._clock()
            bucket = self._buckets.setdefault(user, _Bucket(self.capacity, now))
            self._refill(bucket, now)
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return bucket.tokens
            retry_after = (cost - bucket.tokens) / self.rate_per_s
            raise RateLimitExceeded(retry_after)

    async def status(self, user: str) -> dict:
        async with self._lock:
            now = self._clock()
            bucket = self._buckets.setdefault(user, _Bucket(self.capacity, now))
            self._refill(bucket, now)
            return {"tokens": round(bucket.tokens, 2), "capacity": self.capacity,
                    "refill_per_minute": round(self.rate_per_s * 60, 2)}
