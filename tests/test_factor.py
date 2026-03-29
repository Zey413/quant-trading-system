"""因子模块单元测试

覆盖:
- Factor 基类 / FactorRegistry
- 动量因子 (MomentumFactor, ReverseReturnFactor)
- 价值因子 (PEFactor, PBFactor, PSFactor)
- 质量因子 (ROEFactor, ROAFactor)
- 因子分析器 (FactorAnalyzer)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.factor.base import Factor, FactorDirection, FactorRegistry
from quant_trading.factor.momentum import MomentumFactor, ReverseReturnFactor
from quant_trading.factor.value import PEFactor, PBFactor, PSFactor
from quant_trading.factor.quality import ROEFactor, ROAFactor
from quant_trading.factor.analyzer import FactorAnalyzer, FactorAnalysisResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stock_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成模拟个股行情数据"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)

    close = 10 * np.exp(np.cumsum(0.001 + 0.02 * rng.randn(n)))
    return pd.DataFrame(
        {
            "open": close * (1 + 0.005 * rng.randn(n)),
            "high": close * (1 + abs(0.01 * rng.randn(n))),
            "low": close * (1 - abs(0.01 * rng.randn(n))),
            "close": close,
            "volume": rng.randint(100000, 1000000, n).astype(float),
            "pe_ttm": 10 + 5 * rng.rand(n),
            "pb": 1 + 2 * rng.rand(n),
            "ps_ttm": 2 + 3 * rng.rand(n),
            "roe_ttm": 0.05 + 0.15 * rng.rand(n),
            "roa_ttm": 0.02 + 0.08 * rng.rand(n),
        },
        index=dates,
    )


def _make_panel_data(
    n_stocks: int = 30,
    n_dates: int = 50,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """生成截面面板数据，用于因子分析

    Returns
    -------
    tuple
        (panel_data, factor_df, return_df)
        panel_data: {symbol: df}
        factor_df: 因子值面板 (dates x stocks)
        return_df: 收益率面板 (dates x stocks)
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    symbols = [f"stock_{i:03d}" for i in range(n_stocks)]

    panel = {}
    factor_vals = {}
    return_vals = {}

    for sym in symbols:
        close = 10 * np.exp(np.cumsum(0.001 + 0.02 * rng.randn(n_dates)))
        df = pd.DataFrame({"close": close}, index=dates)
        panel[sym] = df

        # 动量因子 = 20日收益率
        momentum = pd.Series(close, index=dates) / pd.Series(close, index=dates).shift(20) - 1
        factor_vals[sym] = momentum

        # 未来收益（随机+因子线性关系+噪声）
        future_ret = 0.002 * momentum + 0.01 * rng.randn(n_dates)
        return_vals[sym] = pd.Series(future_ret, index=dates)

    factor_df = pd.DataFrame(factor_vals)
    return_df = pd.DataFrame(return_vals)

    return panel, factor_df, return_df


@pytest.fixture()
def stock_data():
    return _make_stock_data()


@pytest.fixture()
def panel_data():
    return _make_panel_data()


# ===========================================================================
# Factor 基类 / FactorRegistry
# ===========================================================================

class TestFactorBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Factor()

    def test_concrete_factor_has_name(self, stock_data):
        f = MomentumFactor(window=10)
        assert f.name == "momentum_10"
        assert f.direction == FactorDirection.POSITIVE

    def test_repr(self):
        f = MomentumFactor(window=20)
        r = repr(f)
        assert "MomentumFactor" in r
        assert "momentum_20" in r


class TestFactorRegistry:
    def test_momentum_registered(self):
        names = FactorRegistry.list_factors()
        assert "momentum" in names

    def test_get_factor(self):
        f = FactorRegistry.get("momentum", window=10)
        assert isinstance(f, MomentumFactor)
        assert f.get_params()["window"] == 10

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="未找到因子"):
            FactorRegistry.get("nonexistent_factor_xyz")

    def test_all_builtin_registered(self):
        names = FactorRegistry.list_factors()
        expected = {"momentum", "reverse_return", "pe", "pb", "ps", "roe", "roa"}
        assert expected.issubset(set(names))

    def test_register_non_factor_raises(self):
        with pytest.raises(TypeError, match="只能注册 Factor 的子类"):
            @FactorRegistry.register("bad")
            class NotAFactor:
                pass


# ===========================================================================
# 动量因子
# ===========================================================================

class TestMomentumFactor:
    def test_compute_shape(self, stock_data):
        f = MomentumFactor(window=20)
        result = f.compute(stock_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(stock_data)

    def test_first_n_nan(self, stock_data):
        window = 20
        f = MomentumFactor(window=window)
        result = f.compute(stock_data)
        # 前 window 个值应为 NaN
        assert result.iloc[:window].isna().all()
        # 之后应有有效值
        assert result.iloc[window:].notna().all()

    def test_momentum_value(self):
        """手动验证动量值"""
        dates = pd.bdate_range("2023-01-01", periods=5)
        df = pd.DataFrame(
            {"close": [100.0, 105.0, 110.0, 108.0, 115.0]},
            index=dates,
        )
        f = MomentumFactor(window=2)
        result = f.compute(df)
        # 第3天 (idx=2): 110/100 - 1 = 0.1
        assert abs(result.iloc[2] - 0.1) < 1e-10
        # 第4天 (idx=3): 108/105 - 1 ≈ 0.02857
        assert abs(result.iloc[3] - (108 / 105 - 1)) < 1e-10

    def test_get_params(self):
        f = MomentumFactor(window=30)
        assert f.get_params() == {"window": 30}


class TestReverseReturnFactor:
    def test_is_negative_momentum(self, stock_data):
        mom = MomentumFactor(window=5).compute(stock_data)
        rev = ReverseReturnFactor(window=5).compute(stock_data)

        valid = mom.dropna().index
        np.testing.assert_array_almost_equal(
            rev[valid].values, -mom[valid].values
        )

    def test_direction_positive(self):
        f = ReverseReturnFactor(window=5)
        assert f.direction == FactorDirection.POSITIVE


# ===========================================================================
# 价值因子
# ===========================================================================

class TestPEFactor:
    def test_ep_inverse(self, stock_data):
        """默认 EP = 1/PE"""
        f = PEFactor(inverse=True)
        result = f.compute(stock_data)
        assert result.notna().all()
        # PE > 0 => EP > 0
        assert (result > 0).all()

    def test_pe_direct(self, stock_data):
        f = PEFactor(inverse=False)
        result = f.compute(stock_data)
        assert f.direction == FactorDirection.NEGATIVE
        np.testing.assert_array_almost_equal(
            result.values, stock_data["pe_ttm"].values
        )

    def test_pe_with_negative(self):
        """PE <= 0 时 EP 应为 NaN"""
        dates = pd.bdate_range("2023-01-01", periods=5)
        df = pd.DataFrame(
            {"pe_ttm": [10.0, -5.0, 0.0, 20.0, 15.0]},
            index=dates,
        )
        f = PEFactor(inverse=True)
        result = f.compute(df)
        assert result.iloc[0] == pytest.approx(0.1)
        assert np.isnan(result.iloc[1])  # PE < 0
        assert np.isnan(result.iloc[2])  # PE = 0
        assert result.iloc[3] == pytest.approx(0.05)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        f = PEFactor()
        with pytest.raises(KeyError, match="未找到 PE 列"):
            f.compute(df)


class TestPBFactor:
    def test_bp_inverse(self, stock_data):
        f = PBFactor(inverse=True)
        result = f.compute(stock_data)
        assert result.notna().all()
        assert (result > 0).all()

    def test_pb_direct(self, stock_data):
        f = PBFactor(inverse=False)
        result = f.compute(stock_data)
        np.testing.assert_array_almost_equal(
            result.values, stock_data["pb"].values
        )

    def test_missing_column_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        f = PBFactor()
        with pytest.raises(KeyError, match="未找到 PB 列"):
            f.compute(df)


class TestPSFactor:
    def test_sp_inverse(self, stock_data):
        f = PSFactor(inverse=True)
        result = f.compute(stock_data)
        assert result.notna().all()
        assert (result > 0).all()

    def test_missing_column_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        f = PSFactor()
        with pytest.raises(KeyError, match="未找到 PS 列"):
            f.compute(df)


# ===========================================================================
# 质量因子
# ===========================================================================

class TestROEFactor:
    def test_from_column(self, stock_data):
        f = ROEFactor(ttm=True)
        result = f.compute(stock_data)
        assert result.notna().all()
        np.testing.assert_array_almost_equal(
            result.values, stock_data["roe_ttm"].values
        )

    def test_from_computed(self):
        """从 net_profit / equity 计算"""
        dates = pd.bdate_range("2023-01-01", periods=4)
        df = pd.DataFrame(
            {
                "net_profit": [100, 200, -50, 150],
                "equity": [1000, 1000, 1000, 0],
            },
            index=dates,
        )
        f = ROEFactor()
        result = f.compute(df)
        assert result.iloc[0] == pytest.approx(0.1)
        assert result.iloc[1] == pytest.approx(0.2)
        assert result.iloc[2] == pytest.approx(-0.05)
        # equity = 0 时 ROE 应为 NaN
        assert np.isnan(result.iloc[3])

    def test_missing_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        f = ROEFactor()
        with pytest.raises(KeyError, match="未找到 ROE 相关列"):
            f.compute(df)

    def test_direction(self):
        f = ROEFactor()
        assert f.direction == FactorDirection.POSITIVE


class TestROAFactor:
    def test_from_column(self, stock_data):
        f = ROAFactor(ttm=True)
        result = f.compute(stock_data)
        assert result.notna().all()

    def test_from_computed(self):
        dates = pd.bdate_range("2023-01-01", periods=3)
        df = pd.DataFrame(
            {
                "net_profit": [100, 200, 150],
                "total_assets": [5000, 5000, 0],
            },
            index=dates,
        )
        f = ROAFactor()
        result = f.compute(df)
        assert result.iloc[0] == pytest.approx(0.02)
        assert result.iloc[1] == pytest.approx(0.04)
        assert np.isnan(result.iloc[2])  # total_assets = 0

    def test_missing_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        f = ROAFactor()
        with pytest.raises(KeyError, match="未找到 ROA 相关列"):
            f.compute(df)


# ===========================================================================
# cross_section 计算
# ===========================================================================

class TestCrossSection:
    def test_compute_cross_section(self):
        panel = {}
        dates = pd.bdate_range("2023-01-01", periods=30)
        for i in range(5):
            close = 10 + np.arange(30) * (0.1 * (i + 1))
            panel[f"stock_{i}"] = pd.DataFrame({"close": close}, index=dates)

        f = MomentumFactor(window=5)
        result = f.compute_cross_section(panel)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 5
        assert len(result) == 30


# ===========================================================================
# FactorAnalyzer
# ===========================================================================

class TestFactorAnalyzer:
    def test_analyze_returns_result(self, panel_data):
        _, factor_df, return_df = panel_data

        # 去掉 NaN 行 (前20行是动量窗口)
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        assert isinstance(result, FactorAnalysisResult)
        assert result.factor_name == "momentum_20"

    def test_ic_series_length(self, panel_data):
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        # IC 序列应有数据
        assert len(result.ic_series) > 0
        assert len(result.rank_ic_series) > 0

    def test_ic_range(self, panel_data):
        """IC 值应在 [-1, 1] 范围内"""
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        assert (result.ic_series.abs() <= 1.0 + 1e-10).all()
        assert (result.rank_ic_series.abs() <= 1.0 + 1e-10).all()

    def test_quantile_returns_shape(self, panel_data):
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        n_q = 5
        analyzer = FactorAnalyzer(f, n_quantiles=n_q, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        assert result.quantile_returns.shape[1] == n_q

    def test_coverage(self, panel_data):
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        assert 0.0 <= result.factor_coverage <= 1.0

    def test_autocorrelation_range(self, panel_data):
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        assert -1.0 <= result.factor_autocorr <= 1.0

    def test_monotonicity_range(self, panel_data):
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        assert -1.0 <= result.monotonicity_score <= 1.0

    def test_summary_output(self, panel_data):
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        summary = result.summary()
        assert "因子分析报告" in summary
        assert "IC 均值" in summary
        assert "Rank IC" in summary
        assert "多空年化收益" in summary

    def test_insufficient_data_raises(self):
        """数据不足时应抛出异常"""
        dates = pd.bdate_range("2023-01-01", periods=1)
        factor_df = pd.DataFrame({"A": [1.0]}, index=dates)
        return_df = pd.DataFrame({"A": [0.01]}, index=dates)

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5)

        with pytest.raises(ValueError, match="数据不足"):
            analyzer.analyze(factor_df, return_df)

    def test_quantile_annualized(self, panel_data):
        _, factor_df, return_df = panel_data
        factor_clean = factor_df.iloc[20:]
        return_clean = return_df.iloc[20:]

        f = MomentumFactor(window=20)
        analyzer = FactorAnalyzer(f, n_quantiles=5, holding_period=1)
        result = analyzer.analyze(factor_clean, return_clean)

        assert len(result.quantile_annualized) == 5


# ===========================================================================
# IC / Rank IC 内部方法
# ===========================================================================

class TestICCalculation:
    def test_perfect_positive_ic(self):
        """完美正相关因子的 IC 应为 1"""
        dates = pd.bdate_range("2023-01-01", periods=3)
        stocks = [f"s{i}" for i in range(10)]
        rng = np.random.RandomState(42)

        factor_df = pd.DataFrame(
            rng.randn(3, 10), index=dates, columns=stocks
        )
        # 收益率完全等于因子值
        return_df = factor_df.copy()

        ic = FactorAnalyzer._calc_ic_series(factor_df, return_df)
        assert all(abs(v - 1.0) < 1e-10 for v in ic.values)

    def test_perfect_negative_ic(self):
        """完美负相关因子的 IC 应为 -1"""
        dates = pd.bdate_range("2023-01-01", periods=3)
        stocks = [f"s{i}" for i in range(10)]
        rng = np.random.RandomState(42)

        factor_df = pd.DataFrame(
            rng.randn(3, 10), index=dates, columns=stocks
        )
        return_df = -factor_df

        ic = FactorAnalyzer._calc_ic_series(factor_df, return_df)
        assert all(abs(v + 1.0) < 1e-10 for v in ic.values)


# ===========================================================================
# AssetType 枚举
# ===========================================================================

class TestAssetTypeEnum:
    def test_import(self):
        from quant_trading.core.enums import AssetType
        assert AssetType.STOCK.value == "stock"
        assert AssetType.INDEX.value == "index"
        assert AssetType.ETF.value == "etf"
        assert AssetType.FUTURES.value == "futures"
        assert AssetType.OPTION.value == "option"

    def test_from_value(self):
        from quant_trading.core.enums import AssetType
        assert AssetType("stock") == AssetType.STOCK
        assert AssetType("index") == AssetType.INDEX
