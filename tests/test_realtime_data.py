"""实时行情模块单元测试

测试覆盖:
- RealtimeQuote 模型字段校验与属性计算
- RealtimeQuoteProvider 单股/批量/市场快照获取
- 轮询订阅与取消订阅
- 缓存机制
- 异常处理与边界条件
- 不依赖真实 API 调用（全部 mock）
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from quant_trading.data.realtime import (
    RealtimeQuote,
    RealtimeQuoteProvider,
    _SPOT_COLUMN_MAP,
    _INDEX_CODES,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def sample_spot_df() -> pd.DataFrame:
    """模拟 AKShare stock_zh_a_spot_em 返回的实时行情 DataFrame"""
    return pd.DataFrame(
        {
            "代码": ["000001", "600519", "300750"],
            "名称": ["平安银行", "贵州茅台", "宁德时代"],
            "最新价": [13.72, 1800.00, 220.50],
            "今开": [13.60, 1795.00, 218.00],
            "最高": [13.85, 1810.00, 225.00],
            "最低": [13.50, 1790.00, 216.00],
            "昨收": [13.65, 1805.00, 219.00],
            "成交量": [1200000, 50000, 800000],
            "成交额": [16500000.0, 90000000.0, 176000000.0],
            "涨跌幅": [0.51, -0.28, 0.68],
            "涨跌额": [0.07, -5.00, 1.50],
            "换手率": [1.02, 0.40, 0.65],
            "总市值": [266000000000.0, 2260000000000.0, 512000000000.0],
            "流通市值": [260000000000.0, 2260000000000.0, 480000000000.0],
            "市盈率-动态": [7.50, 30.20, 45.60],
            "市净率": [0.85, 10.50, 5.20],
        }
    )


@pytest.fixture()
def sample_index_df() -> pd.DataFrame:
    """模拟 AKShare stock_zh_index_spot_em 返回的指数行情 DataFrame"""
    return pd.DataFrame(
        {
            "代码": ["000001", "399001", "399006", "000300", "000905", "000688"],
            "名称": [
                "上证指数",
                "深证成指",
                "创业板指",
                "沪深300",
                "中证500",
                "科创50",
            ],
            "最新价": [3250.00, 10800.00, 2200.00, 3900.00, 5600.00, 1050.00],
            "涨跌幅": [0.50, 0.30, 0.80, 0.45, 0.60, 0.70],
            "涨跌额": [16.00, 32.00, 17.00, 17.00, 33.00, 7.00],
            "成交量": [350000000, 450000000, 120000000, 250000000, 180000000, 50000000],
            "成交额": [
                450000000000.0,
                560000000000.0,
                150000000000.0,
                350000000000.0,
                200000000000.0,
                60000000000.0,
            ],
            "今开": [3240.00, 10780.00, 2190.00, 3895.00, 5580.00, 1045.00],
            "最高": [3265.00, 10850.00, 2220.00, 3920.00, 5620.00, 1060.00],
            "最低": [3235.00, 10770.00, 2185.00, 3880.00, 5570.00, 1040.00],
            "昨收": [3234.00, 10768.00, 2183.00, 3883.00, 5567.00, 1043.00],
        }
    )


@pytest.fixture()
def mock_provider(sample_spot_df) -> RealtimeQuoteProvider:
    """创建一个使用 mock 数据的 Provider"""
    provider = RealtimeQuoteProvider(cache_ttl=0)
    return provider


# ======================================================================
# Tests: RealtimeQuote Model
# ======================================================================


class TestRealtimeQuoteModel:
    """RealtimeQuote 数据模型测试"""

    def test_create_basic_quote(self):
        """创建基本行情数据"""
        quote = RealtimeQuote(
            symbol="000001",
            name="平安银行",
            price=13.72,
            open=13.60,
            high=13.85,
            low=13.50,
        )
        assert quote.symbol == "000001"
        assert quote.name == "平安银行"
        assert quote.price == 13.72

    def test_symbol_required(self):
        """symbol 是必填字段"""
        with pytest.raises(Exception):
            RealtimeQuote()

    def test_symbol_cannot_be_empty(self):
        """symbol 不能为空字符串"""
        with pytest.raises(ValueError, match="symbol 不能为空"):
            RealtimeQuote(symbol="")

    def test_symbol_strip_whitespace(self):
        """symbol 应去除前后空白"""
        quote = RealtimeQuote(symbol="  000001  ")
        assert quote.symbol == "000001"

    def test_default_values(self):
        """默认值应为零或空"""
        quote = RealtimeQuote(symbol="000001")
        assert quote.name == ""
        assert quote.price == 0.0
        assert quote.volume == 0
        assert quote.amount == 0.0
        assert quote.bid_price == 0.0
        assert quote.ask_price == 0.0

    def test_is_trading_property(self):
        """is_trading 属性判断"""
        quote_active = RealtimeQuote(symbol="000001", price=13.72)
        quote_inactive = RealtimeQuote(symbol="000002", price=0.0)
        assert quote_active.is_trading is True
        assert quote_inactive.is_trading is False

    def test_amplitude_property(self):
        """振幅计算"""
        quote = RealtimeQuote(
            symbol="000001",
            high=13.85,
            low=13.50,
            prev_close=13.65,
        )
        expected = (13.85 - 13.50) / 13.65 * 100
        assert abs(quote.amplitude - expected) < 0.01

    def test_amplitude_zero_prev_close(self):
        """昨收为0时振幅应为0"""
        quote = RealtimeQuote(
            symbol="000001",
            high=13.85,
            low=13.50,
            prev_close=0.0,
        )
        assert quote.amplitude == 0.0

    def test_to_dict(self):
        """转换为字典"""
        quote = RealtimeQuote(symbol="000001", price=13.72, name="平安银行")
        d = quote.to_dict()
        assert isinstance(d, dict)
        assert d["symbol"] == "000001"
        assert d["price"] == 13.72
        assert d["name"] == "平安银行"

    def test_timestamp_default(self):
        """timestamp 默认为当前时间"""
        before = datetime.now()
        quote = RealtimeQuote(symbol="000001")
        after = datetime.now()
        assert before <= quote.timestamp <= after

    def test_all_fields_populated(self):
        """所有字段均可设置"""
        quote = RealtimeQuote(
            symbol="000001",
            name="平安银行",
            price=13.72,
            open=13.60,
            high=13.85,
            low=13.50,
            prev_close=13.65,
            volume=1200000,
            amount=16500000.0,
            bid_price=13.71,
            ask_price=13.73,
            bid_volume=500,
            ask_volume=800,
            pct_change=0.51,
            price_change=0.07,
            turnover_rate=1.02,
            total_market_cap=266000000000.0,
            float_market_cap=260000000000.0,
            pe_ratio=7.50,
            pb_ratio=0.85,
        )
        assert quote.bid_price == 13.71
        assert quote.ask_volume == 800
        assert quote.pe_ratio == 7.50


# ======================================================================
# Tests: RealtimeQuoteProvider - 单股查询
# ======================================================================


class TestGetRealtimeQuote:
    """单股实时行情查询测试"""

    def test_get_quote_success(self, sample_spot_df):
        """成功获取单股行情"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            quote = provider.get_realtime_quote("000001")

            assert quote.symbol == "000001"
            assert quote.name == "平安银行"
            assert quote.price == 13.72
            assert quote.high == 13.85
            assert quote.low == 13.50

    def test_get_quote_not_found(self, sample_spot_df):
        """股票代码不存在应抛出 ValueError"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            with pytest.raises(ValueError, match="未找到股票"):
                provider.get_realtime_quote("999999")

    def test_get_quote_empty_symbol(self):
        """空 symbol 应抛出 ValueError"""
        provider = RealtimeQuoteProvider()
        with pytest.raises(ValueError, match="symbol 不能为空"):
            provider.get_realtime_quote("")

    def test_get_quote_api_failure(self):
        """API 调用失败应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.side_effect = Exception("Network error")
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            with pytest.raises(RuntimeError, match="获取实时行情失败"):
                provider.get_realtime_quote("000001")

    def test_get_quote_empty_response(self):
        """API 返回空数据应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = pd.DataFrame()
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            with pytest.raises(RuntimeError, match="实时行情数据为空"):
                provider.get_realtime_quote("000001")


# ======================================================================
# Tests: RealtimeQuoteProvider - 批量查询
# ======================================================================


class TestGetRealtimeQuotes:
    """批量实时行情查询测试"""

    def test_batch_success(self, sample_spot_df):
        """批量获取成功"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            quotes = provider.get_realtime_quotes(["000001", "600519"])

            assert len(quotes) == 2
            assert quotes[0].symbol == "000001"
            assert quotes[1].symbol == "600519"

    def test_batch_partial_found(self, sample_spot_df):
        """部分股票未找到时跳过"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            quotes = provider.get_realtime_quotes(
                ["000001", "999999", "600519"]
            )
            assert len(quotes) == 2

    def test_batch_empty_list(self):
        """空列表应抛出 ValueError"""
        provider = RealtimeQuoteProvider()
        with pytest.raises(ValueError, match="symbols 列表不能为空"):
            provider.get_realtime_quotes([])

    def test_batch_single_api_call(self, sample_spot_df):
        """批量查询只应发起一次 API 调用"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            provider.get_realtime_quotes(["000001", "600519", "300750"])

            # 只调用一次 API
            assert mock_ak.stock_zh_a_spot_em.call_count == 1


# ======================================================================
# Tests: RealtimeQuoteProvider - 市场快照
# ======================================================================


class TestGetMarketSnapshot:
    """市场快照测试"""

    def test_snapshot_success(self, sample_index_df):
        """成功获取市场快照"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_index_spot_em.return_value = sample_index_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider()
            snapshot = provider.get_market_snapshot()

            assert "上证指数" in snapshot
            assert "深证成指" in snapshot
            assert "创业板指" in snapshot
            assert snapshot["上证指数"]["price"] == 3250.00
            assert snapshot["沪深300"]["pct_change"] == 0.45

    def test_snapshot_has_expected_fields(self, sample_index_df):
        """快照应包含标准字段"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_index_spot_em.return_value = sample_index_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider()
            snapshot = provider.get_market_snapshot()

            for index_name, data in snapshot.items():
                assert "code" in data
                assert "price" in data
                assert "pct_change" in data
                assert "volume" in data
                assert "timestamp" in data

    def test_snapshot_api_failure(self):
        """API 失败应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_index_spot_em.side_effect = Exception("Timeout")
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider()
            with pytest.raises(RuntimeError, match="获取指数行情失败"):
                provider.get_market_snapshot()

    def test_snapshot_empty_returns_empty_dict(self):
        """空数据返回空字典"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_index_spot_em.return_value = pd.DataFrame()
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider()
            snapshot = provider.get_market_snapshot()
            assert snapshot == {}


# ======================================================================
# Tests: RealtimeQuoteProvider - 订阅
# ======================================================================


class TestSubscription:
    """轮询订阅测试"""

    def test_subscribe_and_receive(self, sample_spot_df):
        """订阅后应收到回调"""
        received: list[RealtimeQuote] = []

        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            provider.subscribe(
                ["000001"],
                callback=lambda q: received.append(q),
                interval=0.1,
            )

            # 等待至少一次回调
            time.sleep(0.3)
            provider.unsubscribe()

            assert len(received) >= 1
            assert received[0].symbol == "000001"

    def test_unsubscribe(self, sample_spot_df):
        """取消订阅后应停止回调"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            provider.subscribe(
                ["000001"],
                callback=lambda q: None,
                interval=0.1,
            )
            assert provider.is_subscribed is True

            provider.unsubscribe()
            assert provider.is_subscribed is False

    def test_double_subscribe_raises(self, sample_spot_df):
        """重复订阅应抛出 ValueError"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=0)
            provider.subscribe(["000001"], callback=lambda q: None, interval=0.1)

            try:
                with pytest.raises(ValueError, match="已有活跃的订阅"):
                    provider.subscribe(
                        ["600519"], callback=lambda q: None, interval=0.1
                    )
            finally:
                provider.unsubscribe()

    def test_subscribe_empty_symbols(self):
        """空 symbols 应抛出 ValueError"""
        provider = RealtimeQuoteProvider()
        with pytest.raises(ValueError, match="symbols 列表不能为空"):
            provider.subscribe([], callback=lambda q: None)

    def test_unsubscribe_when_not_subscribed(self):
        """未订阅时 unsubscribe 不应报错"""
        provider = RealtimeQuoteProvider()
        provider.unsubscribe()  # 不应抛出异常
        assert provider.is_subscribed is False


# ======================================================================
# Tests: RealtimeQuoteProvider - 缓存
# ======================================================================


class TestProviderCache:
    """行情缓存机制测试"""

    def test_cache_prevents_duplicate_calls(self, sample_spot_df):
        """缓存应避免短时间内重复 API 调用"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=10)  # 10秒缓存

            provider.get_realtime_quote("000001")
            provider.get_realtime_quote("600519")

            # 两次调用应只触发一次 API 请求（缓存命中）
            assert mock_ak.stock_zh_a_spot_em.call_count == 1

    def test_clear_cache(self, sample_spot_df):
        """清除缓存后应重新请求"""
        with patch(
            "quant_trading.data.realtime._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_spot_df
            mock_import.return_value = mock_ak

            provider = RealtimeQuoteProvider(cache_ttl=10)

            provider.get_realtime_quote("000001")
            provider.clear_cache()
            provider.get_realtime_quote("000001")

            assert mock_ak.stock_zh_a_spot_em.call_count == 2


# ======================================================================
# Tests: 列名映射与常量
# ======================================================================


class TestConstants:
    """常量和映射测试"""

    def test_spot_column_map_completeness(self):
        """列名映射应包含主要字段"""
        assert "代码" in _SPOT_COLUMN_MAP
        assert "名称" in _SPOT_COLUMN_MAP
        assert "最新价" in _SPOT_COLUMN_MAP
        assert "涨跌幅" in _SPOT_COLUMN_MAP

    def test_index_codes(self):
        """指数代码映射应包含主要指数"""
        assert "上证指数" in _INDEX_CODES
        assert "深证成指" in _INDEX_CODES
        assert "创业板指" in _INDEX_CODES
        assert "沪深300" in _INDEX_CODES

    def test_repr(self):
        """repr 格式"""
        provider = RealtimeQuoteProvider(cache_ttl=5)
        r = repr(provider)
        assert "RealtimeQuoteProvider" in r
        assert "cache_ttl=5" in r
