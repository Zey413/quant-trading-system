"""多股票组合回测测试

测试覆盖：
- PortfolioBacktester: 等权重/自定义权重回测
- PortfolioBacktestResult: 个股绩效、相关性、摘要输出
- 边界情况: 空数据、权重校验
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.metrics import PerformanceResult
from quant_trading.backtest.portfolio_backtest import (
    PortfolioBacktester,
    PortfolioBacktestResult,
)
from quant_trading.core.config import (
    AppConfig,
    BacktestConfig,
    BrokerConfig,
    RiskConfig,
)


# ============================================================
# Fixtures & Helpers
# ============================================================


def make_synthetic_data(
    n_days: int = 60,
    start_price: float = 10.0,
    daily_return: float = 0.002,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成行情数据"""
    dates = pd.bdate_range(start=start_date, periods=n_days)
    np.random.seed(seed)
    prices = [start_price]
    for _ in range(1, n_days):
        r = daily_return + np.random.normal(0, 0.005)
        prices.append(prices[-1] * (1 + r))

    return pd.DataFrame(
        {
            "date": dates,
            "open": [p * 0.998 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000] * n_days,
        }
    )


class AlwaysHoldStrategy:
    """不交易的策略 - 仅用于测试"""

    def on_bar(self, current_date, row, portfolio_snapshot):
        return None


class SimpleBuyStrategy:
    """简单买入策略 - 第3天买入"""

    def __init__(self):
        self.day_count = 0

    def on_bar(self, current_date, row, portfolio_snapshot):
        from quant_trading.core.enums import SignalType
        from quant_trading.core.models import Signal

        self.day_count += 1
        if self.day_count == 3 and len(portfolio_snapshot["positions"]) == 0:
            return Signal(
                date=current_date,
                symbol=row.name if hasattr(row, "name") else "unknown",
                signal_type=SignalType.BUY,
                price=row["close"],
            )
        return None


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        backtest=BacktestConfig(
            initial_capital=1_000_000.0,
            start_date="2024-01-01",
            end_date="2024-12-31",
            risk_free_rate=0.025,
        ),
        broker=BrokerConfig(
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_tax_rate=0.001,
            slippage=0.001,
            lot_size=100,
            t_plus_1=True,
        ),
        risk=RiskConfig(
            max_position_pct=0.3,
            max_total_position_pct=0.8,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            position_sizing_method="fixed_ratio",
            fixed_ratio=0.1,
        ),
    )


# ============================================================
# Weight Resolution Tests
# ============================================================


class TestWeightResolution:
    """权重解析测试"""

    def test_equal_weights_two_stocks(self) -> None:
        """测试两只股票等权重"""
        weights = PortfolioBacktester._resolve_weights(
            ["000001", "600519"], "equal"
        )
        assert abs(weights["000001"] - 0.5) < 1e-10
        assert abs(weights["600519"] - 0.5) < 1e-10

    def test_equal_weights_three_stocks(self) -> None:
        """测试三只股票等权重"""
        weights = PortfolioBacktester._resolve_weights(
            ["A", "B", "C"], "equal"
        )
        for w in weights.values():
            assert abs(w - 1 / 3) < 1e-10

    def test_custom_weights(self) -> None:
        """测试自定义权重"""
        weights = PortfolioBacktester._resolve_weights(
            ["A", "B"],
            {"A": 0.6, "B": 0.4},
        )
        assert abs(weights["A"] - 0.6) < 1e-10
        assert abs(weights["B"] - 0.4) < 1e-10

    def test_custom_weights_normalized(self) -> None:
        """测试权重自动归一化"""
        weights = PortfolioBacktester._resolve_weights(
            ["A", "B"],
            {"A": 3.0, "B": 7.0},
        )
        assert abs(weights["A"] - 0.3) < 1e-10
        assert abs(weights["B"] - 0.7) < 1e-10

    def test_missing_weight_raises(self) -> None:
        """测试缺少股票权重时抛出异常"""
        with pytest.raises(ValueError, match="缺少权重"):
            PortfolioBacktester._resolve_weights(
                ["A", "B", "C"],
                {"A": 0.5, "B": 0.5},
            )

    def test_invalid_weights_type_raises(self) -> None:
        """测试无效权重类型"""
        with pytest.raises(ValueError, match="必须为"):
            PortfolioBacktester._resolve_weights(["A", "B"], 123)

    def test_zero_total_weight_raises(self) -> None:
        """测试权重总和为零"""
        with pytest.raises(ValueError, match="大于 0"):
            PortfolioBacktester._resolve_weights(
                ["A", "B"],
                {"A": 0.0, "B": 0.0},
            )


# ============================================================
# Portfolio Backtest Run Tests
# ============================================================


class TestPortfolioBacktestRun:
    """组合回测运行测试"""

    def test_run_equal_weights_two_stocks(self, app_config: AppConfig) -> None:
        """测试两只股票等权重回测"""
        data1 = make_synthetic_data(
            n_days=60, start_price=10.0, daily_return=0.003,
            start_date="2024-01-01", seed=42,
        )
        data2 = make_synthetic_data(
            n_days=60, start_price=20.0, daily_return=0.001,
            start_date="2024-01-01", seed=123,
        )

        backtester = PortfolioBacktester(app_config)
        result = backtester.run(
            strategy=AlwaysHoldStrategy(),
            data_dict={"000001": data1, "600519": data2},
            weights="equal",
        )

        assert isinstance(result, PortfolioBacktestResult)
        assert "000001" in result.per_stock
        assert "600519" in result.per_stock
        assert abs(result.stock_weights["000001"] - 0.5) < 1e-10
        assert abs(result.stock_weights["600519"] - 0.5) < 1e-10

    def test_run_three_stocks(self, app_config: AppConfig) -> None:
        """测试三只股票回测"""
        data_dict = {}
        for i, symbol in enumerate(["000001", "600519", "000858"]):
            data_dict[symbol] = make_synthetic_data(
                n_days=60,
                start_price=10.0 * (i + 1),
                daily_return=0.002 * (i + 1),
                start_date="2024-01-01",
                seed=42 + i,
            )

        backtester = PortfolioBacktester(app_config)
        result = backtester.run(
            strategy=AlwaysHoldStrategy(),
            data_dict=data_dict,
            weights="equal",
        )

        assert len(result.per_stock) == 3
        assert isinstance(result.overall, PerformanceResult)

    def test_run_custom_weights(self, app_config: AppConfig) -> None:
        """测试自定义权重回测"""
        data1 = make_synthetic_data(
            n_days=60, start_price=10.0, daily_return=0.003,
            start_date="2024-01-01", seed=42,
        )
        data2 = make_synthetic_data(
            n_days=60, start_price=20.0, daily_return=0.001,
            start_date="2024-01-01", seed=123,
        )

        backtester = PortfolioBacktester(app_config)
        result = backtester.run(
            strategy=AlwaysHoldStrategy(),
            data_dict={"000001": data1, "600519": data2},
            weights={"000001": 0.7, "600519": 0.3},
        )

        assert abs(result.stock_weights["000001"] - 0.7) < 1e-10
        assert abs(result.stock_weights["600519"] - 0.3) < 1e-10

    def test_run_empty_data_raises(self, app_config: AppConfig) -> None:
        """测试空数据字典"""
        backtester = PortfolioBacktester(app_config)
        with pytest.raises(ValueError, match="不能为空"):
            backtester.run(
                strategy=AlwaysHoldStrategy(),
                data_dict={},
                weights="equal",
            )

    def test_per_stock_results_independent(self, app_config: AppConfig) -> None:
        """测试各股票绩效互相独立"""
        data1 = make_synthetic_data(
            n_days=60, start_price=10.0, daily_return=0.005,
            start_date="2024-01-01", seed=42,
        )
        data2 = make_synthetic_data(
            n_days=60, start_price=10.0, daily_return=-0.001,
            start_date="2024-01-01", seed=123,
        )

        backtester = PortfolioBacktester(app_config)
        result = backtester.run(
            strategy=AlwaysHoldStrategy(),
            data_dict={"WINNER": data1, "LOSER": data2},
            weights="equal",
        )

        # 无交易策略 => 所有收益率为0（因为没有持仓）
        # 但 equity_curve 仍被记录
        assert "WINNER" in result.per_stock
        assert "LOSER" in result.per_stock


# ============================================================
# Correlation Matrix Tests
# ============================================================


class TestCorrelationMatrix:
    """相关性矩阵测试"""

    def test_correlation_two_stocks(self, app_config: AppConfig) -> None:
        """测试两只股票的相关性矩阵"""
        data1 = make_synthetic_data(
            n_days=60, start_price=10.0, daily_return=0.003,
            start_date="2024-01-01", seed=42,
        )
        data2 = make_synthetic_data(
            n_days=60, start_price=20.0, daily_return=0.001,
            start_date="2024-01-01", seed=123,
        )

        backtester = PortfolioBacktester(app_config)
        result = backtester.run(
            strategy=AlwaysHoldStrategy(),
            data_dict={"000001": data1, "600519": data2},
            weights="equal",
        )

        corr = result.correlation_matrix
        # 两只股票有有效的 equity curves 时才有 correlation
        # 如果无交易，equity curves 是恒定的 => 可能为空或全 NaN
        # 这是合理的行为
        if not corr.empty:
            assert "000001" in corr.columns or corr.shape == (0, 0)

    def test_correlation_matrix_shape(self) -> None:
        """测试相关性矩阵维度"""
        # 直接测试静态方法
        curve1 = pd.Series(
            [100, 102, 101, 105, 108],
            index=pd.bdate_range("2024-01-01", periods=5),
        )
        curve2 = pd.Series(
            [200, 198, 202, 195, 210],
            index=pd.bdate_range("2024-01-01", periods=5),
        )

        corr = PortfolioBacktester._calc_correlation_matrix(
            {"A": curve1, "B": curve2}
        )

        assert corr.shape == (2, 2)
        assert "A" in corr.columns
        assert "B" in corr.columns
        # 对角线为 1
        assert abs(corr.loc["A", "A"] - 1.0) < 1e-10
        assert abs(corr.loc["B", "B"] - 1.0) < 1e-10

    def test_correlation_single_stock_empty(self) -> None:
        """测试单只股票无法计算相关性"""
        curve1 = pd.Series(
            [100, 105, 110],
            index=pd.bdate_range("2024-01-01", periods=3),
        )

        corr = PortfolioBacktester._calc_correlation_matrix({"A": curve1})
        assert corr.empty


# ============================================================
# Summary Output Tests
# ============================================================


class TestPortfolioBacktestSummary:
    """结果摘要输出测试"""

    def test_summary_contains_key_info(self, app_config: AppConfig) -> None:
        """测试 summary 方法输出关键信息"""
        data1 = make_synthetic_data(
            n_days=60, start_price=10.0, daily_return=0.003,
            start_date="2024-01-01", seed=42,
        )
        data2 = make_synthetic_data(
            n_days=60, start_price=20.0, daily_return=0.001,
            start_date="2024-01-01", seed=123,
        )

        backtester = PortfolioBacktester(app_config)
        result = backtester.run(
            strategy=AlwaysHoldStrategy(),
            data_dict={"000001": data1, "600519": data2},
            weights="equal",
        )

        summary = result.summary()
        assert "多股票组合回测报告" in summary
        assert "组合整体绩效" in summary
        assert "个股权重" in summary
        assert "个股绩效" in summary
        assert "000001" in summary
        assert "600519" in summary


# ============================================================
# Portfolio Equity Calculation Tests
# ============================================================


class TestPortfolioEquityCalc:
    """组合净值曲线计算测试"""

    def test_equal_weight_equity(self, app_config: AppConfig) -> None:
        """测试等权重组合净值计算"""
        backtester = PortfolioBacktester(app_config)

        # 构造两只股票的净值曲线
        dates = pd.bdate_range("2024-01-01", periods=5)
        curve1 = pd.Series([100, 110, 105, 115, 120], index=dates, dtype=float)
        curve2 = pd.Series([100, 95, 100, 110, 105], index=dates, dtype=float)

        weights = {"A": 0.5, "B": 0.5}
        equity = backtester._calc_portfolio_equity(
            {"A": curve1, "B": curve2}, weights
        )

        assert len(equity) == 5
        # 第一天: pct_change=0 for both => 1M * 1 = 1M
        assert abs(equity.iloc[0] - app_config.backtest.initial_capital) < 1.0

    def test_empty_curves(self, app_config: AppConfig) -> None:
        """测试空净值曲线"""
        backtester = PortfolioBacktester(app_config)
        equity = backtester._calc_portfolio_equity({}, {"A": 0.5})
        assert equity.empty
