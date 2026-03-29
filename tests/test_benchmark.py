"""BenchmarkManager 单元测试"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.benchmark import (
    INDEX_CODE_MAP,
    BenchmarkManager,
    BenchmarkResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dates(n: int = 244, start: str = "2023-01-03") -> pd.DatetimeIndex:
    """生成 n 个工作日日期"""
    return pd.bdate_range(start=start, periods=n)


def _make_price_series(
    dates: pd.DatetimeIndex,
    start_price: float = 100.0,
    annual_return: float = 0.10,
    volatility: float = 0.20,
    seed: int = 42,
) -> pd.Series:
    """生成带漂移的模拟价格序列"""
    rng = np.random.RandomState(seed)
    n = len(dates)
    daily_drift = annual_return / 244
    daily_vol = volatility / np.sqrt(244)
    log_returns = daily_drift + daily_vol * rng.randn(n)
    log_returns[0] = 0  # 第一天无收益
    prices = start_price * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture()
def dates():
    return _make_dates(244)


@pytest.fixture()
def strategy_curve(dates):
    """策略净值曲线（年化约15%）"""
    return _make_price_series(dates, annual_return=0.15, seed=42)


@pytest.fixture()
def benchmark_data(dates):
    """基准（沪深300）模拟数据（年化约8%）"""
    return _make_price_series(dates, annual_return=0.08, seed=99)


# ---------------------------------------------------------------------------
# 测试 INDEX_CODE_MAP
# ---------------------------------------------------------------------------

class TestIndexCodeMap:
    def test_hs300_chinese(self):
        assert INDEX_CODE_MAP["沪深300"] == "sh000300"

    def test_hs300_english(self):
        assert INDEX_CODE_MAP["hs300"] == "sh000300"

    def test_sz50(self):
        assert INDEX_CODE_MAP["上证50"] == "sh000016"

    def test_zz500(self):
        assert INDEX_CODE_MAP["zz500"] == "sh000905"


# ---------------------------------------------------------------------------
# 测试 BenchmarkManager 初始化
# ---------------------------------------------------------------------------

class TestBenchmarkManagerInit:
    def test_init_with_series(self, benchmark_data):
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        assert bm._benchmark_close is not None
        assert len(bm._benchmark_close) == len(benchmark_data)

    def test_init_with_dataframe(self, dates, benchmark_data):
        df = pd.DataFrame({"close": benchmark_data.values}, index=dates)
        bm = BenchmarkManager("test", benchmark_data=df)
        assert bm._benchmark_close is not None

    def test_init_with_dataframe_capital_close(self, dates, benchmark_data):
        df = pd.DataFrame({"Close": benchmark_data.values}, index=dates)
        bm = BenchmarkManager("test", benchmark_data=df)
        assert bm._benchmark_close is not None

    def test_init_without_data(self):
        bm = BenchmarkManager("沪深300")
        assert bm._benchmark_close is None

    def test_benchmark_close_raises_without_data(self):
        bm = BenchmarkManager("test")
        with pytest.raises(ValueError, match="基准数据未加载"):
            _ = bm.benchmark_close

    def test_invalid_dataframe_raises(self, dates):
        df = pd.DataFrame({"price": [1, 2, 3]}, index=dates[:3])
        with pytest.raises(ValueError, match="必须包含"):
            BenchmarkManager("test", benchmark_data=df)


# ---------------------------------------------------------------------------
# 测试 resolve_index_code
# ---------------------------------------------------------------------------

class TestResolveIndexCode:
    def test_chinese_name(self):
        assert BenchmarkManager.resolve_index_code("沪深300") == "sh000300"

    def test_english_name(self):
        assert BenchmarkManager.resolve_index_code("hs300") == "sh000300"

    def test_unknown_passes_through(self):
        assert BenchmarkManager.resolve_index_code("custom_index") == "custom_index"


# ---------------------------------------------------------------------------
# 测试 compare
# ---------------------------------------------------------------------------

class TestBenchmarkCompare:
    def test_compare_returns_result(self, strategy_curve, benchmark_data):
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(strategy_curve, risk_free_rate=0.03)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "test"

    def test_alpha_positive_for_outperforming_strategy(
        self, strategy_curve, benchmark_data
    ):
        """策略年化15% > 基准8%，Alpha 应为正"""
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(strategy_curve, risk_free_rate=0.03)
        # Alpha 不一定严格为正（取决于 Beta），但超额收益应为正
        assert result.excess_return > 0

    def test_beta_range(self, strategy_curve, benchmark_data):
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(strategy_curve)
        # Beta 应在合理范围内
        assert -3.0 < result.beta < 3.0

    def test_excess_curve_length(self, strategy_curve, benchmark_data):
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(strategy_curve)
        # 超额收益曲线应有数据
        assert len(result.excess_curve) > 0

    def test_benchmark_curve_starts_at_one(self, strategy_curve, benchmark_data):
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(strategy_curve)
        assert abs(result.benchmark_curve.iloc[0] - 1.0) < 1e-10

    def test_tracking_error_positive(self, strategy_curve, benchmark_data):
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(strategy_curve)
        assert result.tracking_error >= 0

    def test_identical_curves_zero_excess(self, benchmark_data):
        """相同曲线对比，超额收益为0"""
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(benchmark_data)
        assert abs(result.excess_return) < 1e-8
        assert abs(result.alpha) < 1e-8
        assert abs(result.beta - 1.0) < 1e-6

    def test_summary_output(self, strategy_curve, benchmark_data):
        bm = BenchmarkManager("test", benchmark_data=benchmark_data)
        result = bm.compare(strategy_curve)
        summary = result.summary()
        assert "基准比较报告" in summary
        assert "Alpha" in summary
        assert "Beta" in summary
        assert "信息比率" in summary


# ---------------------------------------------------------------------------
# 测试 _calc 静态方法
# ---------------------------------------------------------------------------

class TestCalcMethods:
    def test_calc_beta_zero_variance(self):
        """基准无波动时 Beta 应为 0"""
        strat = pd.Series([0.01, -0.02, 0.015, 0.005])
        bench = pd.Series([0.0, 0.0, 0.0, 0.0])
        beta = BenchmarkManager._calc_beta(strat, bench)
        assert beta == 0.0

    def test_calc_beta_perfect_correlation(self):
        """完全正相关时 Beta > 0"""
        returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        beta = BenchmarkManager._calc_beta(returns, returns)
        assert abs(beta - 1.0) < 1e-10

    def test_calc_alpha_capm(self):
        """CAPM: Alpha = Rp - [Rf + Beta * (Rm - Rf)]"""
        alpha = BenchmarkManager._calc_alpha(
            strat_annualized=0.15,
            bench_annualized=0.10,
            beta=1.0,
            risk_free_rate=0.03,
        )
        # Alpha = 0.15 - [0.03 + 1.0 * (0.10 - 0.03)] = 0.15 - 0.10 = 0.05
        assert abs(alpha - 0.05) < 1e-10

    def test_calc_tracking_error_zero(self):
        """无超额收益时跟踪误差为 0"""
        excess = np.zeros(100)
        te = BenchmarkManager._calc_tracking_error(excess)
        assert te == 0.0

    def test_calc_information_ratio_zero_te(self):
        ir = BenchmarkManager._calc_information_ratio(0.05, 0.0)
        assert ir == 0.0

    def test_calc_information_ratio_normal(self):
        ir = BenchmarkManager._calc_information_ratio(0.05, 0.10)
        assert abs(ir - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_short_series(self):
        """极短的序列不应崩溃"""
        dates = pd.bdate_range("2023-01-01", periods=3)
        strat = pd.Series([100, 101, 102], index=dates)
        bench = pd.Series([100, 100.5, 101], index=dates)

        bm = BenchmarkManager("test", benchmark_data=bench)
        result = bm.compare(strat)
        assert isinstance(result, BenchmarkResult)

    def test_misaligned_dates(self):
        """日期不完全重叠也能工作"""
        dates1 = pd.bdate_range("2023-01-01", periods=100)
        dates2 = pd.bdate_range("2023-01-15", periods=100)

        strat = _make_price_series(dates1, seed=1)
        bench = _make_price_series(dates2, seed=2)

        bm = BenchmarkManager("test", benchmark_data=bench)
        result = bm.compare(strat)
        assert isinstance(result, BenchmarkResult)
