"""速率限制器 - 令牌桶算法与指数退避重试

提供:
- RateLimiter: 基于令牌桶算法的速率限制器，线程安全
- @rate_limited(rps=5): 装饰器，为函数调用添加速率限制
- 指数退避重试机制，自动处理临时性网络错误

用法::

    from quant_trading.data.rate_limiter import RateLimiter, rate_limited

    # 方式1: 直接使用 RateLimiter
    limiter = RateLimiter(rate=5.0, burst=10)
    limiter.acquire()  # 阻塞直到令牌可用
    do_api_call()

    # 方式2: 装饰器
    @rate_limited(rps=5)
    def fetch_data(symbol: str) -> dict:
        return api.get(symbol)

    # 方式3: 带重试的装饰器
    @rate_limited(rps=5, max_retries=3, base_delay=1.0)
    def fetch_data_with_retry(symbol: str) -> dict:
        return api.get(symbol)
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RateLimiter:
    """令牌桶速率限制器 - 线程安全

    使用令牌桶算法控制请求速率。桶以固定速率补充令牌，
    每次请求消耗一个令牌。当桶为空时，请求将阻塞等待。

    Parameters
    ----------
    rate : float
        令牌补充速率（每秒补充的令牌数），即允许的 RPS
    burst : int
        令牌桶容量（突发请求上限）。默认等于 rate 的整数值。
        允许短时间内的突发请求，但长期速率仍受 rate 限制。

    Attributes
    ----------
    rate : float
        每秒令牌补充速率
    burst : int
        桶容量（最大突发令牌数）
    tokens : float
        当前可用令牌数
    last_refill : float
        上次令牌补充的时间戳

    Examples
    --------
    >>> limiter = RateLimiter(rate=5.0, burst=10)
    >>> limiter.acquire()        # 消耗1个令牌，必要时阻塞
    >>> limiter.try_acquire()    # 非阻塞尝试，返回 bool
    True
    """

    def __init__(self, rate: float = 5.0, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"rate 必须大于 0，当前值: {rate}")
        self.rate: float = rate
        self.burst: int = burst if burst is not None else max(1, int(rate))
        self.tokens: float = float(self.burst)
        self.last_refill: float = time.monotonic()
        self._lock = threading.Lock()
        self._total_acquired: int = 0
        self._total_waited: float = 0.0
        logger.debug(
            "RateLimiter 初始化: rate=%.1f rps, burst=%d",
            self.rate,
            self.burst,
        )

    def _refill(self) -> None:
        """补充令牌（必须在锁内调用）"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        if new_tokens > 0:
            self.tokens = min(self.burst, self.tokens + new_tokens)
            self.last_refill = now

    def acquire(self, timeout: float | None = None) -> bool:
        """获取一个令牌（阻塞）

        Parameters
        ----------
        timeout : float | None
            最大等待时间（秒）。None 表示无限等待。

        Returns
        -------
        bool
            True 表示成功获取令牌，False 表示超时

        Raises
        ------
        TimeoutError
            当 timeout 不为 None 且等待超时时抛出
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self._total_acquired += 1
                    return True

                # 计算需要等待的时间
                wait_time = (1.0 - self.tokens) / self.rate

            # 检查超时
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("RateLimiter.acquire 超时")
                    return False
                wait_time = min(wait_time, remaining)

            logger.debug("RateLimiter 等待 %.3f 秒", wait_time)
            self._total_waited += wait_time
            time.sleep(wait_time)

    def try_acquire(self) -> bool:
        """非阻塞地尝试获取一个令牌

        Returns
        -------
        bool
            True 表示成功获取，False 表示当前无可用令牌
        """
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self._total_acquired += 1
                return True
            return False

    def wait_time(self) -> float:
        """获取下一个令牌可用前需等待的时间（秒）

        Returns
        -------
        float
            等待时间。0.0 表示立即可用。
        """
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                return 0.0
            return (1.0 - self.tokens) / self.rate

    def reset(self) -> None:
        """重置令牌桶到初始状态"""
        with self._lock:
            self.tokens = float(self.burst)
            self.last_refill = time.monotonic()
            self._total_acquired = 0
            self._total_waited = 0.0
        logger.debug("RateLimiter 已重置")

    def stats(self) -> dict[str, Any]:
        """返回统计信息

        Returns
        -------
        dict[str, Any]
            包含 rate, burst, tokens, total_acquired, total_waited_seconds
        """
        with self._lock:
            self._refill()
            return {
                "rate": self.rate,
                "burst": self.burst,
                "tokens": round(self.tokens, 2),
                "total_acquired": self._total_acquired,
                "total_waited_seconds": round(self._total_waited, 3),
            }

    def __repr__(self) -> str:
        return (
            f"RateLimiter(rate={self.rate}, burst={self.burst}, "
            f"tokens={self.tokens:.1f})"
        )


# ======================================================================
# 指数退避重试
# ======================================================================

# 默认可重试的异常类型
_DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    RuntimeError,
)


def retry_with_backoff(
    func: Callable[..., Any] | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[..., Any]:
    """指数退避重试装饰器

    每次重试的等待时间按 base_delay * 2^attempt 指数增长，
    可选添加随机抖动以避免"惊群效应"。

    Parameters
    ----------
    func : Callable | None
        被装饰的函数（无参调用时为 None）
    max_retries : int
        最大重试次数（不含首次调用）
    base_delay : float
        基础等待时间（秒）
    max_delay : float
        最大等待时间上限（秒）
    jitter : bool
        是否添加随机抖动
    retryable_exceptions : tuple[type[Exception], ...] | None
        可重试的异常类型。None 使用默认列表。

    Returns
    -------
    Callable
        包装后的函数

    Examples
    --------
    >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
    ... def unstable_api_call():
    ...     return requests.get("https://api.example.com")
    """
    exceptions = retryable_exceptions or _DEFAULT_RETRYABLE_EXCEPTIONS

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt >= max_retries:
                        logger.error(
                            "%s 在 %d 次重试后仍然失败: %s",
                            fn.__name__,
                            max_retries,
                            exc,
                        )
                        raise
                    # 计算退避时间
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    logger.warning(
                        "%s 第 %d 次重试（共 %d 次），等待 %.2f 秒: %s",
                        fn.__name__,
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            # 理论上不会到这里，但为了类型安全
            raise last_exception  # type: ignore[misc]

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ======================================================================
# 组合装饰器: 速率限制 + 重试
# ======================================================================

# 全局默认限速器实例（模块级别共享）
_default_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def _get_or_create_limiter(rps: float, burst: int | None = None) -> RateLimiter:
    """获取或创建指定 RPS 的全局限速器

    对于同一 rps 值，返回相同的限速器实例。
    不同 rps 值会创建新实例。
    """
    global _default_limiter
    with _limiter_lock:
        if _default_limiter is None or _default_limiter.rate != rps:
            _default_limiter = RateLimiter(rate=rps, burst=burst)
        return _default_limiter


def rate_limited(
    rps: float = 5.0,
    burst: int | None = None,
    max_retries: int = 0,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    limiter: RateLimiter | None = None,
) -> Callable[[F], F]:
    """速率限制装饰器（可选带重试）

    为函数调用添加速率限制。每次调用前会先获取令牌。
    可选启用指数退避重试。

    Parameters
    ----------
    rps : float
        每秒请求数限制
    burst : int | None
        突发请求上限。None 表示等于 rps。
    max_retries : int
        最大重试次数。0 表示不重试。
    base_delay : float
        重试基础等待时间（秒）
    max_delay : float
        最大重试等待时间（秒）
    jitter : bool
        重试时是否添加随机抖动
    limiter : RateLimiter | None
        自定义限速器实例。None 使用内部全局实例。

    Returns
    -------
    Callable
        装饰器函数

    Examples
    --------
    >>> @rate_limited(rps=5)
    ... def fetch_quote(symbol: str) -> dict:
    ...     return api.get_quote(symbol)

    >>> @rate_limited(rps=3, max_retries=3, base_delay=2.0)
    ... def fetch_with_retry(symbol: str) -> dict:
    ...     return api.get_quote(symbol)
    """

    def decorator(func: F) -> F:
        # 使用传入的 limiter 或创建/获取默认的
        _limiter = limiter or _get_or_create_limiter(rps, burst)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _limiter.acquire()
            return func(*args, **kwargs)

        # 若需重试，叠加重试装饰器
        if max_retries > 0:
            wrapper = retry_with_backoff(
                wrapper,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter=jitter,
            )

        return wrapper  # type: ignore[return-value]

    return decorator
