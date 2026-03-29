"""动量指标单元测试

使用合成数据验证 ROC、WilliamsR、CCI 的计算正确性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.indicators.momentum import CCI, ROC, WilliamsR


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
# ROC
# =========================================================================

class TestROC:
    def test_known_values(self):
        """使用已知数据验证 ROC 计算结果。"""
        data = pd.DataFrame({"close": [100.0, 110.0, 120.0, 130.0, 140.0]})
        roc = ROC(period=2)
        result = roc.calculate(data)
        # ROC(2) at index 2: (120-100)/100*100 = 20.0
        # ROC(2) at index 3: (130-110)/110*100 ≈ 18.18
        # ROC(2) at index 4: (140-120)/120*100 ≈ 16.67
        assert np.isnan(result["roc_2"].iloc[0])
        assert np.isnan(result["roc_2"].iloc[1])
        np.testing.assert_almost_equal(result["roc_2"].iloc[2], 20.0)
        np.testing.assert_almost_equal(
            result["roc_2"].iloc[3], (130 - 110) / 110 * 100, decimal=4
        )
        np.testing.assert_almost_equal(
            result["roc_2"].iloc[4], (140 - 120) / 120 * 100, decimal=4
        )

    def test_period_1(self):
        """period=1 的 ROC 等于日收益率 * 100。"""
        data = pd.DataFrame({"close": [100.0, 105.0, 110.0]})
        result = ROC(period=1).calculate(data)
        assert np.isnan(result["roc_1"].iloc[0])
        np.testing.assert_almost_equal(result["roc_1"].iloc[1], 5.0)
        np.testing.assert_almost_equal(
            result["roc_1"].iloc[2], (110 - 105) / 105 * 100, decimal=4
        )

    def test_column_name(self):
        roc = ROC(period=12)
        assert roc.name == "roc_12"

    def test_required_columns(self):
        assert ROC(period=5).required_columns == ["close"]

    def test_missing_column_raises(self):
        data = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少必需列"):
            ROC(period=1).calculate(data)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            ROC(period=0)

    def test_flat_price_roc_zero(self):
        """恒定价格的 ROC 应为 0。"""
        data = pd.DataFrame({"close": [100.0] * 20})
        result = ROC(period=5).calculate(data)
        valid = result["roc_5"].dropna()
        np.testing.assert_array_almost_equal(valid.to_numpy(), np.zeros(len(valid)))

    def test_with_synthetic_data(self):
        """使用合成数据确保不报错。"""
        data = _make_ohlcv(n=100)
        result = ROC(period=12).calculate(data)
        assert "roc_12" in result.columns
        assert result["roc_12"].dropna().shape[0] > 0


# =========================================================================
# WilliamsR
# =========================================================================

class TestWilliamsR:
    def test_value_range(self):
        """Williams %R 值应在 [-100, 0] 范围内。"""
        data = _make_ohlcv(n=100, seed=7)
        result = WilliamsR(period=14).calculate(data)
        wr = result["williams_r_14"].dropna()
        assert wr.min() >= -100.0
        assert wr.max() <= 0.0

    def test_known_values(self):
        """使用已知高低收盘价验证。"""
        # 简单情况：5 日窗口，第5天 close 等于 highest_high
        data = pd.DataFrame({
            "high":  [10.0, 12.0, 14.0, 13.0, 14.0],
            "low":   [8.0,  9.0,  10.0, 11.0, 12.0],
            "close": [9.0,  11.0, 13.0, 12.0, 14.0],
        })
        result = WilliamsR(period=5).calculate(data)
        # HH = 14, LL = 8, close = 14
        # %R = (14-14)/(14-8)*-100 = 0
        np.testing.assert_almost_equal(result["williams_r_5"].iloc[4], 0.0)

    def test_close_at_lowest(self):
        """收盘价等于最低低点时 %R 应为 -100。"""
        data = pd.DataFrame({
            "high":  [10.0, 12.0, 14.0, 13.0, 11.0],
            "low":   [8.0,  9.0,  10.0, 11.0, 8.0],
            "close": [9.0,  11.0, 13.0, 12.0, 8.0],
        })
        result = WilliamsR(period=5).calculate(data)
        # HH = 14, LL = 8, close = 8
        # %R = (14-8)/(14-8)*-100 = -100
        np.testing.assert_almost_equal(result["williams_r_5"].iloc[4], -100.0)

    def test_column_name(self):
        wr = WilliamsR(period=21)
        assert wr.name == "williams_r_21"

    def test_required_columns(self):
        assert WilliamsR(period=14).required_columns == ["high", "low", "close"]

    def test_missing_column_raises(self):
        data = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少必需列"):
            WilliamsR(period=2).calculate(data)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            WilliamsR(period=0)

    def test_nan_before_window(self):
        """窗口期之前应为 NaN。"""
        data = _make_ohlcv(n=30)
        result = WilliamsR(period=14).calculate(data)
        assert result["williams_r_14"].iloc[:13].isna().all()


# =========================================================================
# CCI
# =========================================================================

class TestCCI:
    def test_adds_column(self):
        """应正确添加 CCI 列。"""
        data = _make_ohlcv(n=50)
        result = CCI(period=20).calculate(data)
        assert "cci_20" in result.columns
        assert result["cci_20"].dropna().shape[0] > 0

    def test_nan_before_window(self):
        """窗口期之前应为 NaN。"""
        data = _make_ohlcv(n=50)
        result = CCI(period=20).calculate(data)
        assert result["cci_20"].iloc[:19].isna().all()

    def test_flat_market_cci_near_zero(self):
        """恒定价格市场中 CCI 应为 0。"""
        n = 50
        data = pd.DataFrame({
            "high": np.full(n, 10.0),
            "low": np.full(n, 10.0),
            "close": np.full(n, 10.0),
        })
        result = CCI(period=20).calculate(data)
        valid = result["cci_20"].dropna()
        # CCI should be 0 when prices don't change
        np.testing.assert_array_almost_equal(
            valid.to_numpy(), np.zeros(len(valid))
        )

    def test_known_trend(self):
        """单调上涨序列中 CCI 应为正值。"""
        n = 60
        prices = np.arange(1, n + 1, dtype=float)
        data = pd.DataFrame({
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
        })
        result = CCI(period=20).calculate(data)
        valid = result["cci_20"].dropna()
        # 在单调上涨中 CCI 应为正值
        assert valid.iloc[-1] > 0

    def test_column_name(self):
        cci = CCI(period=14)
        assert cci.name == "cci_14"

    def test_required_columns(self):
        assert CCI(period=20).required_columns == ["high", "low", "close"]

    def test_missing_column_raises(self):
        data = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少必需列"):
            CCI(period=2).calculate(data)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            CCI(period=0)

    def test_with_synthetic_data(self):
        """使用合成数据确保不报错且范围合理。"""
        data = _make_ohlcv(n=200, seed=99)
        result = CCI(period=20).calculate(data)
        valid = result["cci_20"].dropna()
        # CCI 通常在 -300 ~ +300 范围内，极端不超过 -500 ~ +500
        assert valid.max() < 1000
        assert valid.min() > -1000


# =========================================================================
# 跨动量指标集成测试
# =========================================================================

class TestMomentumIntegration:
    def test_chained_momentum_indicators(self):
        """多个动量指标可以依次叠加到同一个 DataFrame 上。"""
        data = _make_ohlcv(n=100)
        ROC(period=12).calculate(data)
        WilliamsR(period=14).calculate(data)
        CCI(period=20).calculate(data)

        expected_cols = {"roc_12", "williams_r_14", "cci_20"}
        assert expected_cols.issubset(set(data.columns))
