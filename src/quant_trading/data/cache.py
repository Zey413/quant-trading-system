"""数据缓存模块 - LRU内存缓存 + SQLite磁盘缓存

提供:
- DataCache: 双层缓存（内存 LRU + SQLite 持久化），线程安全
- @cached(ttl=300): 函数结果缓存装饰器，支持 TTL 过期
- CacheStats: 缓存统计信息

用法::

    from quant_trading.data.cache import DataCache, cached

    # 方式1: 直接使用 DataCache
    cache = DataCache(max_memory_items=1000, db_path="cache.db")
    cache.set("key1", {"price": 13.5}, ttl=300)
    value = cache.get("key1")

    # 方式2: 装饰器
    @cached(ttl=300)
    def get_stock_info(symbol: str) -> dict:
        return api.fetch(symbol)
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheEntry:
    """缓存条目

    Attributes
    ----------
    value : Any
        缓存的值
    expire_at : float
        过期时间戳（time.monotonic）。0 表示永不过期。
    created_at : float
        创建时间戳
    """

    value: Any
    expire_at: float
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        """检查是否已过期"""
        if self.expire_at == 0:
            return False
        return time.monotonic() > self.expire_at


@dataclass
class CacheStats:
    """缓存统计信息

    Attributes
    ----------
    memory_hits : int
        内存缓存命中次数
    disk_hits : int
        磁盘缓存命中次数
    misses : int
        缓存未命中次数
    memory_size : int
        内存缓存当前条目数
    disk_size : int
        磁盘缓存当前条目数
    evictions : int
        LRU 淘汰次数
    """

    memory_hits: int = 0
    disk_hits: int = 0
    misses: int = 0
    memory_size: int = 0
    disk_size: int = 0
    evictions: int = 0

    @property
    def total_hits(self) -> int:
        """总命中次数"""
        return self.memory_hits + self.disk_hits

    @property
    def total_requests(self) -> int:
        """总请求次数"""
        return self.total_hits + self.misses

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "memory_hits": self.memory_hits,
            "disk_hits": self.disk_hits,
            "misses": self.misses,
            "memory_size": self.memory_size,
            "disk_size": self.disk_size,
            "evictions": self.evictions,
            "total_hits": self.total_hits,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
        }


class DataCache:
    """双层数据缓存 - LRU内存 + SQLite磁盘

    内存层使用 OrderedDict 实现 LRU 淘汰策略。
    磁盘层使用 SQLite 实现持久化存储。
    两层均支持 TTL 过期机制。

    Parameters
    ----------
    max_memory_items : int
        内存缓存最大条目数
    db_path : str | Path | None
        SQLite 数据库路径。None 表示仅使用内存缓存（":memory:"）。
    default_ttl : int
        默认 TTL（秒）。0 表示永不过期。
    enable_disk : bool
        是否启用磁盘缓存

    Examples
    --------
    >>> cache = DataCache(max_memory_items=500, db_path="data_cache.db")
    >>> cache.set("stock:000001", {"price": 13.5}, ttl=60)
    >>> cache.get("stock:000001")
    {"price": 13.5}
    """

    def __init__(
        self,
        max_memory_items: int = 1024,
        db_path: str | Path | None = None,
        default_ttl: int = 300,
        enable_disk: bool = True,
    ) -> None:
        if max_memory_items <= 0:
            raise ValueError(
                f"max_memory_items 必须大于 0，当前值: {max_memory_items}"
            )
        self._max_memory_items = max_memory_items
        self._default_ttl = default_ttl
        self._enable_disk = enable_disk

        # 内存层: OrderedDict 实现 LRU
        self._memory: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # 统计
        self._stats = CacheStats()

        # 磁盘层: SQLite
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn: sqlite3.Connection | None = None
        if self._enable_disk:
            self._init_db()

        logger.debug(
            "DataCache 初始化: max_memory=%d, db_path=%s, default_ttl=%d",
            max_memory_items,
            self._db_path,
            default_ttl,
        )

    def _init_db(self) -> None:
        """初始化 SQLite 数据库"""
        if self._db_path != ":memory:":
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                expire_at REAL NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_expire ON cache(expire_at)"
        )
        self._conn.commit()

    def _serialize(self, value: Any) -> bytes:
        """序列化值为 bytes"""
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, data: bytes) -> Any:
        """从 bytes 反序列化值"""
        return pickle.loads(data)  # noqa: S301

    def get(self, key: str) -> Any | None:
        """获取缓存值

        先查内存缓存，未命中则查磁盘缓存。
        命中磁盘缓存时会提升到内存缓存。

        Parameters
        ----------
        key : str
            缓存键

        Returns
        -------
        Any | None
            缓存值。未命中或已过期返回 None。
        """
        with self._lock:
            # 1. 查内存
            entry = self._memory.get(key)
            if entry is not None:
                if entry.is_expired():
                    del self._memory[key]
                    logger.debug("内存缓存过期: %s", key)
                else:
                    # LRU: 移到末尾
                    self._memory.move_to_end(key)
                    self._stats.memory_hits += 1
                    return entry.value

            # 2. 查磁盘
            if self._enable_disk and self._conn is not None:
                row = self._conn.execute(
                    "SELECT value, expire_at, created_at FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    value_blob, expire_at, created_at = row
                    # 检查过期（磁盘层使用 time.time）
                    if expire_at > 0 and time.time() > expire_at:
                        self._conn.execute(
                            "DELETE FROM cache WHERE key = ?", (key,)
                        )
                        self._conn.commit()
                        logger.debug("磁盘缓存过期: %s", key)
                    else:
                        value = self._deserialize(value_blob)
                        # 提升到内存缓存
                        self._memory_put(
                            key,
                            CacheEntry(
                                value=value,
                                expire_at=(
                                    time.monotonic() + (expire_at - time.time())
                                    if expire_at > 0
                                    else 0
                                ),
                                created_at=time.monotonic(),
                            ),
                        )
                        self._stats.disk_hits += 1
                        return value

        self._stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """设置缓存值

        同时写入内存缓存和磁盘缓存。

        Parameters
        ----------
        key : str
            缓存键
        value : Any
            缓存值（必须可序列化）
        ttl : int | None
            TTL（秒）。None 使用默认 TTL。0 表示永不过期。
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl

        with self._lock:
            # 内存层
            expire_at = (
                time.monotonic() + effective_ttl if effective_ttl > 0 else 0
            )
            entry = CacheEntry(value=value, expire_at=expire_at)
            self._memory_put(key, entry)

            # 磁盘层
            if self._enable_disk and self._conn is not None:
                disk_expire = (
                    time.time() + effective_ttl if effective_ttl > 0 else 0
                )
                try:
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO cache (key, value, expire_at, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (key, self._serialize(value), disk_expire, time.time()),
                    )
                    self._conn.commit()
                except Exception as exc:
                    logger.error("磁盘缓存写入失败 [%s]: %s", key, exc)

    def _memory_put(self, key: str, entry: CacheEntry) -> None:
        """写入内存缓存并执行 LRU 淘汰（必须在锁内调用）"""
        if key in self._memory:
            self._memory.move_to_end(key)
            self._memory[key] = entry
        else:
            self._memory[key] = entry
            # LRU 淘汰
            while len(self._memory) > self._max_memory_items:
                evicted_key, _ = self._memory.popitem(last=False)
                self._stats.evictions += 1
                logger.debug("LRU 淘汰: %s", evicted_key)
        self._stats.memory_size = len(self._memory)

    def delete(self, key: str) -> bool:
        """删除缓存条目

        Parameters
        ----------
        key : str
            缓存键

        Returns
        -------
        bool
            True 表示成功删除，False 表示键不存在
        """
        deleted = False
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                self._stats.memory_size = len(self._memory)
                deleted = True

            if self._enable_disk and self._conn is not None:
                cursor = self._conn.execute(
                    "DELETE FROM cache WHERE key = ?", (key,)
                )
                self._conn.commit()
                if cursor.rowcount > 0:
                    deleted = True

        return deleted

    def clear(self) -> int:
        """清除所有缓存

        Returns
        -------
        int
            清除的条目总数
        """
        with self._lock:
            count = len(self._memory)
            self._memory.clear()
            self._stats.memory_size = 0

            if self._enable_disk and self._conn is not None:
                cursor = self._conn.execute("SELECT COUNT(*) FROM cache")
                disk_count = cursor.fetchone()[0]
                self._conn.execute("DELETE FROM cache")
                self._conn.commit()
                count += disk_count

        logger.info("缓存已清除，共 %d 条", count)
        return count

    def cleanup_expired(self) -> int:
        """清理所有过期条目

        Returns
        -------
        int
            清理的条目数
        """
        count = 0
        with self._lock:
            # 内存层
            expired_keys = [
                k for k, v in self._memory.items() if v.is_expired()
            ]
            for key in expired_keys:
                del self._memory[key]
                count += 1
            self._stats.memory_size = len(self._memory)

            # 磁盘层
            if self._enable_disk and self._conn is not None:
                cursor = self._conn.execute(
                    "DELETE FROM cache WHERE expire_at > 0 AND expire_at < ?",
                    (time.time(),),
                )
                self._conn.commit()
                count += cursor.rowcount

        if count > 0:
            logger.info("清理过期缓存条目: %d", count)
        return count

    def contains(self, key: str) -> bool:
        """检查键是否存在（且未过期）

        Parameters
        ----------
        key : str
            缓存键

        Returns
        -------
        bool
            键存在且未过期返回 True
        """
        return self.get(key) is not None

    def size(self) -> dict[str, int]:
        """获取缓存大小信息

        Returns
        -------
        dict[str, int]
            包含 memory_count 和 disk_count
        """
        with self._lock:
            memory_count = len(self._memory)
            disk_count = 0
            if self._enable_disk and self._conn is not None:
                cursor = self._conn.execute("SELECT COUNT(*) FROM cache")
                disk_count = cursor.fetchone()[0]
        return {"memory_count": memory_count, "disk_count": disk_count}

    def get_stats(self) -> CacheStats:
        """获取缓存统计信息

        Returns
        -------
        CacheStats
            统计信息数据对象
        """
        with self._lock:
            self._stats.memory_size = len(self._memory)
            if self._enable_disk and self._conn is not None:
                cursor = self._conn.execute("SELECT COUNT(*) FROM cache")
                self._stats.disk_size = cursor.fetchone()[0]
        return self._stats

    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("DataCache 数据库连接已关闭")

    def __del__(self) -> None:
        """析构时关闭连接"""
        self.close()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DataCache(memory={stats.memory_size}, "
            f"disk={stats.disk_size}, "
            f"hit_rate={stats.hit_rate:.1%})"
        )


# ======================================================================
# 缓存装饰器
# ======================================================================

# 模块级别的默认缓存实例
_default_cache: DataCache | None = None
_cache_init_lock = threading.Lock()


def get_default_cache() -> DataCache:
    """获取或创建默认的全局缓存实例

    Returns
    -------
    DataCache
        全局默认缓存实例
    """
    global _default_cache
    with _cache_init_lock:
        if _default_cache is None:
            _default_cache = DataCache(
                max_memory_items=2048,
                enable_disk=False,
            )
    return _default_cache


def set_default_cache(cache: DataCache) -> None:
    """设置默认缓存实例

    Parameters
    ----------
    cache : DataCache
        缓存实例
    """
    global _default_cache
    with _cache_init_lock:
        _default_cache = cache


def _make_cache_key(func: Callable[..., Any], args: tuple, kwargs: dict) -> str:
    """为函数调用生成缓存键

    基于函数全限定名、位置参数和关键字参数生成 SHA256 哈希键。

    Parameters
    ----------
    func : Callable
        函数对象
    args : tuple
        位置参数
    kwargs : dict
        关键字参数

    Returns
    -------
    str
        缓存键字符串
    """
    key_parts = [func.__module__, func.__qualname__]
    for arg in args:
        key_parts.append(repr(arg))
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v!r}")

    raw_key = "|".join(key_parts)
    hashed = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"{func.__qualname__}:{hashed}"


def cached(
    ttl: int = 300,
    cache: DataCache | None = None,
    key_prefix: str = "",
) -> Callable[[F], F]:
    """函数结果缓存装饰器

    自动缓存函数返回值，支持 TTL 过期。
    缓存键基于函数名和参数自动生成。

    Parameters
    ----------
    ttl : int
        缓存有效期（秒）。0 表示永不过期。
    cache : DataCache | None
        自定义缓存实例。None 使用全局默认缓存。
    key_prefix : str
        缓存键前缀，用于区分不同场景。

    Returns
    -------
    Callable
        装饰器函数

    Examples
    --------
    >>> @cached(ttl=60)
    ... def get_price(symbol: str) -> float:
    ...     return api.get_price(symbol)

    >>> # 首次调用: 执行函数并缓存
    >>> price = get_price("000001")
    >>> # 再次调用: 直接返回缓存值
    >>> price = get_price("000001")
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _cache = cache or get_default_cache()
            raw_key = _make_cache_key(func, args, kwargs)
            cache_key = f"{key_prefix}{raw_key}" if key_prefix else raw_key

            # 尝试从缓存获取
            result = _cache.get(cache_key)
            if result is not None:
                logger.debug("缓存命中: %s", cache_key)
                return result

            # 执行函数
            result = func(*args, **kwargs)

            # 写入缓存
            _cache.set(cache_key, result, ttl=ttl)
            logger.debug("缓存写入: %s (ttl=%d)", cache_key, ttl)
            return result

        # 附加工具方法
        wrapper.cache_clear = lambda: (cache or get_default_cache()).clear()  # type: ignore[attr-defined]
        wrapper.cache_key = lambda *a, **kw: _make_cache_key(func, a, kw)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
