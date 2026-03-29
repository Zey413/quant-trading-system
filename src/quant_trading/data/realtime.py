"""实时行情模块 - 基于AKShare的实时行情数据获取

提供:
- RealtimeQuote: 实时行情Pydantic数据模型
- RealtimeQuoteProvider: 实时行情数据提供者，支持单股/批量/市场快照/轮询订阅

用法::

    from quant_trading.data.realtime import RealtimeQuoteProvider

    provider = RealtimeQuoteProvider()
    quote = provider.get_realtime_quote("000001")
    print(f"{quote.name}: {quote.price}")

    # 批量获取
    quotes = provider.get_realtime_quotes(["000001", "600519"])

    # 市场快照
    snapshot = provider.get_market_snapshot()

    # 轮询订阅
    def on_quote(quote):
        print(f"{quote.symbol}: {quote.price}")

    provider.subscribe(["000001", "600519"], callback=on_quote, interval=3)
    # ... 稍后
    provider.unsubscribe()
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ======================================================================
# AKShare 延迟导入
# ======================================================================


def _import_akshare():
    """延迟导入 akshare，并处理未安装的情况"""
    try:
        import akshare as ak
        return ak
    except ImportError:
        raise ImportError(
            "akshare is not installed. Install it with: pip install akshare"
        )


# ======================================================================
# 数据模型
# ======================================================================


class RealtimeQuote(BaseModel):
    """实时行情数据模型

    基于东方财富实时行情接口的标准化数据结构。

    Attributes
    ----------
    symbol : str
        股票代码（纯数字，如 "000001"）
    name : str
        股票名称
    price : float
        最新价
    open : float
        今开盘价
    high : float
        最高价
    low : float
        最低价
    prev_close : float
        昨收盘价
    volume : int
        成交量（手）
    amount : float
        成交额（元）
    bid_price : float
        买一价
    ask_price : float
        卖一价
    bid_volume : int
        买一量（手）
    ask_volume : int
        卖一量（手）
    pct_change : float
        涨跌幅（%）
    price_change : float
        涨跌额
    turnover_rate : float
        换手率（%）
    total_market_cap : float
        总市值（元）
    float_market_cap : float
        流通市值（元）
    pe_ratio : float
        市盈率（动态）
    pb_ratio : float
        市净率
    timestamp : datetime
        行情时间戳
    """

    symbol: str = Field(..., description="股票代码")
    name: str = Field(default="", description="股票名称")
    price: float = Field(default=0.0, description="最新价")
    open: float = Field(default=0.0, description="今开盘价")
    high: float = Field(default=0.0, description="最高价")
    low: float = Field(default=0.0, description="最低价")
    prev_close: float = Field(default=0.0, description="昨收盘价")
    volume: int = Field(default=0, description="成交量（手）")
    amount: float = Field(default=0.0, description="成交额（元）")
    bid_price: float = Field(default=0.0, description="买一价")
    ask_price: float = Field(default=0.0, description="卖一价")
    bid_volume: int = Field(default=0, description="买一量（手）")
    ask_volume: int = Field(default=0, description="卖一量（手）")
    pct_change: float = Field(default=0.0, description="涨跌幅(%)")
    price_change: float = Field(default=0.0, description="涨跌额")
    turnover_rate: float = Field(default=0.0, description="换手率(%)")
    total_market_cap: float = Field(default=0.0, description="总市值（元）")
    float_market_cap: float = Field(default=0.0, description="流通市值（元）")
    pe_ratio: float = Field(default=0.0, description="市盈率（动态）")
    pb_ratio: float = Field(default=0.0, description="市净率")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="行情时间戳",
    )

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        """校验股票代码格式"""
        v = v.strip()
        if not v:
            raise ValueError("symbol 不能为空")
        return v

    @property
    def is_trading(self) -> bool:
        """是否在交易中（价格大于0）"""
        return self.price > 0

    @property
    def amplitude(self) -> float:
        """振幅 (%)"""
        if self.prev_close == 0:
            return 0.0
        return (self.high - self.low) / self.prev_close * 100

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


# ======================================================================
# 东方财富列名映射
# ======================================================================

# AKShare stock_zh_a_spot_em 接口中文列名 -> RealtimeQuote 字段映射
_SPOT_COLUMN_MAP: dict[str, str] = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "price",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "昨收": "prev_close",
    "成交量": "volume",
    "成交额": "amount",
    "买一价": "bid_price",      # 部分接口可能不含此字段
    "卖一价": "ask_price",
    "买一量": "bid_volume",
    "卖一量": "ask_volume",
    "涨跌幅": "pct_change",
    "涨跌额": "price_change",
    "换手率": "turnover_rate",
    "总市值": "total_market_cap",
    "流通市值": "float_market_cap",
    "市盈率-动态": "pe_ratio",
    "市净率": "pb_ratio",
}

# 市场指数代码映射
_INDEX_CODES: dict[str, str] = {
    "上证指数": "000001",
    "深证成指": "399001",
    "创业板指": "399006",
    "科创50": "000688",
    "沪深300": "000300",
    "中证500": "000905",
}


# ======================================================================
# 实时行情提供者
# ======================================================================


class RealtimeQuoteProvider:
    """实时行情数据提供者

    基于 AKShare 的 stock_zh_a_spot_em 接口获取实时行情。
    支持单股查询、批量查询、市场快照和轮询订阅。

    Parameters
    ----------
    cache_ttl : int
        行情缓存TTL（秒），避免短时间内重复请求。默认2秒。

    Notes
    -----
    - AKShare 有访问频率限制，建议 subscribe 的 interval >= 3 秒
    - 轮询订阅在后台线程中运行，通过 unsubscribe() 停止

    Examples
    --------
    >>> provider = RealtimeQuoteProvider()
    >>> quote = provider.get_realtime_quote("000001")
    >>> print(quote.price)
    13.72
    """

    def __init__(self, cache_ttl: int = 2) -> None:
        self._cache_ttl = cache_ttl
        self._spot_cache: dict[str, Any] | None = None
        self._cache_time: float = 0.0
        self._cache_lock = threading.Lock()

        # 订阅相关
        self._subscribe_thread: threading.Thread | None = None
        self._subscribe_stop_event = threading.Event()
        self._is_subscribed: bool = False

        logger.debug("RealtimeQuoteProvider 初始化: cache_ttl=%d", cache_ttl)

    def _fetch_spot_data(self) -> Any:
        """获取全市场实时行情（带缓存）

        Returns
        -------
        pd.DataFrame
            东方财富实时行情 DataFrame

        Raises
        ------
        RuntimeError
            API 调用失败时抛出
        """
        import pandas as pd

        with self._cache_lock:
            now = time.monotonic()
            if (
                self._spot_cache is not None
                and (now - self._cache_time) < self._cache_ttl
            ):
                return self._spot_cache

        ak = _import_akshare()
        try:
            df = ak.stock_zh_a_spot_em()
        except Exception as exc:
            logger.error("获取实时行情失败: %s", exc)
            raise RuntimeError(
                f"获取实时行情失败: {exc}"
            ) from exc

        if df is None or df.empty:
            raise RuntimeError("实时行情数据为空")

        with self._cache_lock:
            self._spot_cache = df
            self._cache_time = time.monotonic()

        return df

    def _row_to_quote(self, row: Any) -> RealtimeQuote:
        """将 DataFrame 行转换为 RealtimeQuote

        Parameters
        ----------
        row : pd.Series
            行情数据行

        Returns
        -------
        RealtimeQuote
            标准化的行情数据对象
        """
        data: dict[str, Any] = {}
        for cn_col, en_field in _SPOT_COLUMN_MAP.items():
            if cn_col in row.index:
                val = row[cn_col]
                # 处理 NaN / None
                if val is None or (isinstance(val, float) and val != val):
                    continue
                data[en_field] = val

        # 确保 symbol 存在
        if "symbol" not in data:
            data["symbol"] = str(row.get("代码", ""))

        data["timestamp"] = datetime.now()
        return RealtimeQuote(**data)

    def get_realtime_quote(self, symbol: str) -> RealtimeQuote:
        """获取单只股票的实时行情

        Parameters
        ----------
        symbol : str
            股票代码（纯数字，如 "000001"）

        Returns
        -------
        RealtimeQuote
            实时行情数据

        Raises
        ------
        ValueError
            股票代码无效或未找到
        RuntimeError
            API 调用失败
        """
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol 不能为空")

        df = self._fetch_spot_data()

        # 筛选目标股票
        code_col = "代码" if "代码" in df.columns else df.columns[1]
        row = df[df[code_col] == symbol]

        if row.empty:
            raise ValueError(f"未找到股票: {symbol}")

        quote = self._row_to_quote(row.iloc[0])
        logger.debug("获取行情: %s (%s) = %.2f", symbol, quote.name, quote.price)
        return quote

    def get_realtime_quotes(self, symbols: list[str]) -> list[RealtimeQuote]:
        """批量获取多只股票的实时行情

        只发起一次 API 调用，然后从结果中筛选。

        Parameters
        ----------
        symbols : list[str]
            股票代码列表

        Returns
        -------
        list[RealtimeQuote]
            行情数据列表。未找到的股票会被跳过（记录警告日志）。

        Raises
        ------
        ValueError
            symbols 为空列表
        RuntimeError
            API 调用失败
        """
        if not symbols:
            raise ValueError("symbols 列表不能为空")

        cleaned = [s.strip() for s in symbols]
        df = self._fetch_spot_data()

        code_col = "代码" if "代码" in df.columns else df.columns[1]
        result: list[RealtimeQuote] = []

        for symbol in cleaned:
            row = df[df[code_col] == symbol]
            if row.empty:
                logger.warning("未找到股票: %s，已跳过", symbol)
                continue
            quote = self._row_to_quote(row.iloc[0])
            result.append(quote)

        logger.debug("批量获取行情: 请求 %d 只，返回 %d 只", len(symbols), len(result))
        return result

    def get_market_snapshot(self) -> dict[str, dict[str, Any]]:
        """获取主要市场指数快照

        返回上证指数、深证成指、创业板指等主要指数的实时数据。

        Returns
        -------
        dict[str, dict[str, Any]]
            键为指数名称，值为指数行情字典。

        Raises
        ------
        RuntimeError
            API 调用失败
        """
        ak = _import_akshare()

        snapshot: dict[str, dict[str, Any]] = {}

        try:
            df = ak.stock_zh_index_spot_em()
        except Exception as exc:
            logger.error("获取指数行情失败: %s", exc)
            raise RuntimeError(f"获取指数行情失败: {exc}") from exc

        if df is None or df.empty:
            logger.warning("指数行情数据为空")
            return snapshot

        # 查找主要指数
        code_col = "代码" if "代码" in df.columns else df.columns[0]
        name_col = "名称" if "名称" in df.columns else df.columns[1]

        for index_name, index_code in _INDEX_CODES.items():
            row = df[df[code_col] == index_code]
            if row.empty:
                logger.warning("未找到指数: %s (%s)", index_name, index_code)
                continue

            r = row.iloc[0]
            snapshot[index_name] = {
                "code": index_code,
                "name": str(r.get(name_col, index_name)),
                "price": float(r.get("最新价", 0)),
                "pct_change": float(r.get("涨跌幅", 0)),
                "price_change": float(r.get("涨跌额", 0)),
                "volume": int(r.get("成交量", 0)),
                "amount": float(r.get("成交额", 0)),
                "open": float(r.get("今开", 0)),
                "high": float(r.get("最高", 0)),
                "low": float(r.get("最低", 0)),
                "prev_close": float(r.get("昨收", 0)),
                "timestamp": datetime.now().isoformat(),
            }

        logger.debug("市场快照: 获取 %d 个指数", len(snapshot))
        return snapshot

    def subscribe(
        self,
        symbols: list[str],
        callback: Callable[[RealtimeQuote], None],
        interval: float = 3.0,
    ) -> None:
        """轮询订阅实时行情

        在后台线程中定期获取行情数据，并通过回调函数推送。

        Parameters
        ----------
        symbols : list[str]
            订阅的股票代码列表
        callback : Callable[[RealtimeQuote], None]
            行情更新回调函数，每只股票的每次更新都会调用一次
        interval : float
            轮询间隔（秒）。建议 >= 3 以避免触发限流。

        Raises
        ------
        ValueError
            symbols 为空或已经在订阅中
        """
        if not symbols:
            raise ValueError("symbols 列表不能为空")

        if self._is_subscribed:
            raise ValueError("已有活跃的订阅，请先调用 unsubscribe()")

        self._subscribe_stop_event.clear()
        self._is_subscribed = True

        def _poll_loop() -> None:
            logger.info(
                "开始订阅行情: symbols=%s, interval=%.1fs",
                symbols,
                interval,
            )
            while not self._subscribe_stop_event.is_set():
                try:
                    quotes = self.get_realtime_quotes(symbols)
                    for quote in quotes:
                        try:
                            callback(quote)
                        except Exception as cb_exc:
                            logger.error(
                                "行情回调异常 [%s]: %s",
                                quote.symbol,
                                cb_exc,
                            )
                except Exception as exc:
                    logger.error("轮询获取行情失败: %s", exc)

                # 使用 Event.wait 支持快速退出
                self._subscribe_stop_event.wait(timeout=interval)

            logger.info("订阅已停止")

        self._subscribe_thread = threading.Thread(
            target=_poll_loop,
            daemon=True,
            name="realtime-quote-subscriber",
        )
        self._subscribe_thread.start()

    def unsubscribe(self) -> None:
        """取消订阅

        停止后台轮询线程。如果没有活跃订阅，调用此方法无副作用。
        """
        if not self._is_subscribed:
            logger.debug("没有活跃的订阅")
            return

        self._subscribe_stop_event.set()
        if self._subscribe_thread is not None:
            self._subscribe_thread.join(timeout=10)
            if self._subscribe_thread.is_alive():
                logger.warning("订阅线程未能在 10 秒内停止")
            self._subscribe_thread = None

        self._is_subscribed = False
        logger.info("已取消订阅")

    @property
    def is_subscribed(self) -> bool:
        """是否有活跃的订阅"""
        return self._is_subscribed

    def clear_cache(self) -> None:
        """清除内部行情缓存"""
        with self._cache_lock:
            self._spot_cache = None
            self._cache_time = 0.0
        logger.debug("实时行情缓存已清除")

    def __repr__(self) -> str:
        return (
            f"RealtimeQuoteProvider(cache_ttl={self._cache_ttl}, "
            f"subscribed={self._is_subscribed})"
        )

    def __del__(self) -> None:
        """析构时自动取消订阅"""
        if self._is_subscribed:
            self.unsubscribe()
