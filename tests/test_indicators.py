"""技术指标单元测试

使用合成数据验证所有指标的计算正确性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.indicators.oscillator import KDJ, RSI
from quant_trading.indicators.trend import EMA, MACD, SMA
from quant_trading.indicators.volatility import ATR, BollingerBands
from quant_trading.indicators.volume import OBV, VWAP


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 100,
    seed: int = 42,
    base_price: float = 10.0,
) -> pd.DataFrame:
    """生成包含 open/high/low/close/volume 的合成行情数据。"""
    rng = np.random.RandomState(seed)
    close = base_price + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(1000, 100_000, n).astype(float)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# =========================================================================
# SMA
# =========================================================================

class TestSMA:
    def test_known_values(self):
        """使用已知数据验证 SMA 计算结果。"""
        data = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        sma = SMA(window=3)
        result = sma.calculate(data)
        expected = [np.nan, np.nan, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(
            result["sma_3"].to_numpy(), expected
        )

    def test_window_1(self):
        """window=1 时 SMA 应等于原值。"""
        data = pd.DataFrame({"close": [10.0, 20.0, 30.0]})
        result = SMA(window=1).calculate(data)
        np.testing.assert_array_almost_equal(
            result["sma_1"].to_numpy(), [10.0, 20.0, 30.0]
        )

    def test_column_name(self):
        sma = SMA(window=5)
        assert sma.name == "sma_5"

    def test_required_columns(self):
        assert SMA(window=3).required_columns == ["close"]

    def test_missing_column_raises(self):
        data = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少必需列"):
            SMA(window=2).calculate(data)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            SMA(window=0)


# =========================================================================
# EMA
# =========================================================================

class TestEMA:
    def test_adds_column(self):
        data = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = EMA(window=3).calculate(data)
        assert "ema_3" in result.columns
        assert len(result["ema_3"].dropna()) == 5  # EMA 从第 1 个值开始有输出

    def test_ema_trend(self):
        """对于单调递增序列，EMA 应单调递增。"""
        data = pd.DataFrame({"close": list(range(1, 21))})
        result = EMA(window=5).calculate(data)
        ema = result["ema_5"].to_numpy()
        diffs = np.diff(ema)
        assert np.all(diffs > 0)


# =========================================================================
# MACD
# =========================================================================

class TestMACD:
    def test_output_columns_exist(self):
        data = _make_ohlcv(n=50)
        result = MACD().calculate(data)
        for col in ("macd", "macd_signal", "macd_hist"):
            assert col in result.columns

    def test_hist_equals_two_times_diff(self):
        data = _make_ohlcv(n=50)
        result = MACD().calculate(data)
        expected_hist = 2.0 * (result["macd"] - result["macd_signal"])
        np.testing.assert_array_almost_equal(
            result["macd_hist"].to_numpy(), expected_hist.to_numpy()
        )

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            MACD(fast=26, slow=12)  # fast >= slow


# =========================================================================
# RSI
# =========================================================================

class TestRSI:
    def test_boundaries(self):
        """RSI 值应始终在 [0, 100] 范围内。"""
        data = _make_ohlcv(n=200, seed=7)
        result = RSI(period=14).calculate(data)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0

    def test_all_up_rsi_near_100(self):
        """纯上涨序列的 RSI 应接近 100。"""
        data = pd.DataFrame({"close": np.arange(1, 51, dtype=float)})
        result = RSI(period=14).calculate(data)
        rsi = result["rsi_14"].dropna()
        assert rsi.iloc[-1] > 95.0

    def test_all_down_rsi_near_0(self):
        """纯下跌序列的 RSI 应接近 0。"""
        data = pd.DataFrame({"close": np.arange(50, 0, -1, dtype=float)})
        result = RSI(period=14).calculate(data)
        rsi = result["rsi_14"].dropna()
        assert rsi.iloc[-1] < 5.0

    def test_column_naming(self):
        data = _make_ohlcv(n=30)
        result = RSI(period=6).calculate(data)
        assert "rsi_6" in result.columns


# =========================================================================
# KDJ
# =========================================================================

class TestKDJ:
    def test_output_columns(self):
        data = _make_ohlcv(n=50)
        result = KDJ().calculate(data)
        for col in ("k", "d", "j"):
            assert col in result.columns

    def test_j_formula(self):
        """J = 3*K - 2*D"""
        data = _make_ohlcv(n=50)
        result = KDJ().calculate(data)
        valid = result.dropna(subset=["k", "d", "j"])
        expected_j = 3.0 * valid["k"] - 2.0 * valid["d"]
        np.testing.assert_array_almost_equal(
            valid["j"].to_numpy(), expected_j.to_numpy()
        )

    def test_k_d_initial_values(self):
        """K 和 D 不应全为 NaN（除前 n-1 行外）。"""
        data = _make_ohlcv(n=30)
        result = KDJ(n=9).calculate(data)
        assert result["k"].dropna().shape[0] > 0
        assert result["d"].dropna().shape[0] > 0


# =========================================================================
# BollingerBands
# =========================================================================

class TestBollingerBands:
    def test_output_columns(self):
        data = _make_ohlcv(n=50)
        result = BollingerBands(window=20).calculate(data)
        for col in ("bb_upper", "bb_middle", "bb_lower", "bb_width"):
            assert col in result.columns

    def test_middle_equals_sma(self):
        """布林带中轨应等于 SMA。"""
        data = _make_ohlcv(n=50)
        result = BollingerBands(window=20).calculate(data)
        sma = SMA(window=20).calculate(data.copy())
        valid_idx = result["bb_middle"].dropna().index
        np.testing.assert_array_almost_equal(
            result.loc[valid_idx, "bb_middle"].to_numpy(),
            sma.loc[valid_idx, "sma_20"].to_numpy(),
        )

    def test_band_ordering(self):
        """上轨 > 中轨 > 下轨。"""
        data = _make_ohlcv(n=50)
        result = BollingerBands(window=20).calculate(data)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_width_nonnegative(self):
        data = _make_ohlcv(n=50)
        result = BollingerBands(window=20).calculate(data)
        valid = result["bb_width"].dropna()
        assert (valid >= 0).all()


# =========================================================================
# ATR
# =========================================================================

class TestATR:
    def test_positive_values(self):
        """ATR 值应始终 >= 0。"""
        data = _make_ohlcv(n=50)
        result = ATR(period=14).calculate(data)
        atr = result["atr_14"].dropna()
        assert (atr >= 0).all()

    def test_column_naming(self):
        data = _make_ohlcv(n=30)
        result = ATR(period=7).calculate(data)
        assert "atr_7" in result.columns

    def test_flat_market_atr(self):
        """在恒定价格市场中，ATR 应趋近于 0。"""
        n = 50
        data = pd.DataFrame({
            "open": np.full(n, 10.0),
            "high": np.full(n, 10.0),
            "low": np.full(n, 10.0),
            "close": np.full(n, 10.0),
            "volume": np.full(n, 1000.0),
        })
        result = ATR(period=14).calculate(data)
        atr = result["atr_14"].dropna()
        # 除第一个 TR (high-low 与 prev_close 的差) 外，其余应趋近 0
        assert atr.iloc[-1] < 0.01


# =========================================================================
# OBV
# =========================================================================

class TestOBV:
    def test_direction_changes(self):
        """OBV 应在价格上涨日增加、下跌日减少。"""
        data = pd.DataFrame({
            "close": [10.0, 11.0, 10.5, 12.0, 11.5],
            "volume": [100.0, 200.0, 150.0, 300.0, 250.0],
        })
        result = OBV().calculate(data)
        obv = result["obv"].to_numpy()
        # day0: 0 (direction=0, obv=0)
        # day1: 涨, obv += 200 -> 200
        # day2: 跌, obv -= 150 -> 50
        # day3: 涨, obv += 300 -> 350
        # day4: 跌, obv -= 250 -> 100
        expected = [0, 200, 50, 350, 100]
        np.testing.assert_array_almost_equal(obv, expected)

    def test_flat_price_obv_unchanged(self):
        """价格不变时 OBV 不变。"""
        data = pd.DataFrame({
            "close": [10.0, 10.0, 10.0],
            "volume": [100.0, 200.0, 300.0],
        })
        result = OBV().calculate(data)
        obv = result["obv"].to_numpy()
        np.testing.assert_array_almost_equal(obv, [0, 0, 0])


# =========================================================================
# VWAP
# =========================================================================

class TestVWAP:
    def test_single_bar(self):
        """单根 K 线的 VWAP 应等于 typical price。"""
        data = pd.DataFrame({
            "high": [12.0],
            "low": [8.0],
            "close": [10.0],
            "volume": [1000.0],
        })
        result = VWAP().calculate(data)
        expected_tp = (12.0 + 8.0 + 10.0) / 3.0
        np.testing.assert_almost_equal(result["vwap"].iloc[0], expected_tp)

    def test_vwap_between_high_low(self):
        """VWAP 应始终在 high 和 low 的范围内（单日内场景）。"""
        data = _make_ohlcv(n=1)
        result = VWAP().calculate(data)
        vwap = result["vwap"].iloc[0]
        assert data["low"].iloc[0] <= vwap <= data["high"].iloc[0]

    def test_zero_volume(self):
        """volume 为 0 时不应出现除以零错误。"""
        data = pd.DataFrame({
            "high": [10.0, 11.0],
            "low": [9.0, 10.0],
            "close": [9.5, 10.5],
            "volume": [0.0, 0.0],
        })
        result = VWAP().calculate(data)
        assert not np.any(np.isnan(result["vwap"].to_numpy()))
        assert not np.any(np.isinf(result["vwap"].to_numpy()))


# =========================================================================
# 跨指标集成测试
# =========================================================================

class TestIntegration:
    def test_chained_indicators(self):
        """多个指标可以依次叠加到同一个 DataFrame 上。"""
        data = _make_ohlcv(n=100)
        SMA(window=10).calculate(data)
        EMA(window=10).calculate(data)
        MACD().calculate(data)
        RSI().calculate(data)
        KDJ().calculate(data)
        BollingerBands().calculate(data)
        ATR().calculate(data)
        OBV().calculate(data)
        VWAP().calculate(data)

        expected_cols = {
            "sma_10", "ema_10",
            "macd", "macd_signal", "macd_hist",
            "rsi_14",
            "k", "d", "j",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr_14",
            "obv", "vwap",
        }
        assert expected_cols.issubset(set(data.columns))
