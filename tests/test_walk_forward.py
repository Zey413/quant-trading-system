"""滚动前进分析测试

测试覆盖：
- WalkForwardAnalyzer: 窗口切分、参数优化、结果拼接
- WalkForwardResult: 汇总输出
- 边界情况: 数据不足、单窗口
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.metrics import PerformanceResult
from quant_trading.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardResult,
    WalkForwardWindow,
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
    n_days: int = 300,
    start_price: float = 10.0,
    daily_return: float = 0.002,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """生成合成行情数据"""
    dates = pd.bdate_range(start=start_date, periods=n_days)
    np.random.seed(42)
    prices = [start_price]
    for _ in range(1, n_days):
        # 加入一些随机波动使信号更真实
        r = daily_return + np.random.normal(0, 0.01)
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


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        backtest=BacktestConfig(
            initial_capital=1_000_000.0,
            start_date="2024-01-01",
            end_date="2025-12-31",
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
# Window Splitting Tests
# ============================================================


class TestWindowSplitting:
    """窗口切分逻辑测试"""

    def test_split_exact_fit(self, app_config: AppConfig) -> None:
        """测试数据恰好整除的窗口切分"""
        analyzer = WalkForwardAnalyzer(app_config)
        dates = pd.bdate_range(start="2024-01-01", periods=240)

        windows = analyzer._split_windows(
            dates, train_size=120, test_size=60, step_size=60
        )

        # 240天, train=120, test=60, step=60
        # 窗口1: 0~119(train), 120~179(test), 下一个 start=60
        # 窗口2: 60~179(train), 180~239(test), 下一个 start=120
        # 窗口3: 120+120+60=300 > 240, 停止
        assert len(windows) == 2

    def test_split_single_window(self, app_config: AppConfig) -> None:
        """测试只够一个窗口的数据"""
        analyzer = WalkForwardAnalyzer(app_config)
        dates = pd.bdate_range(start="2024-01-01", periods=180)

        windows = analyzer._split_windows(
            dates, train_size=120, test_size=60, step_size=60
        )

        assert len(windows) == 1
        train_dates, test_dates = windows[0]
        assert len(train_dates) == 120
        assert len(test_dates) == 60

    def test_split_insufficient_data(self, app_config: AppConfig) -> None:
        """测试数据不足时无窗口"""
        analyzer = WalkForwardAnalyzer(app_config)
        dates = pd.bdate_range(start="2024-01-01", periods=100)

        windows = analyzer._split_windows(
            dates, train_size=120, test_size=60, step_size=60
        )

        assert len(windows) == 0

    def test_split_step_smaller_than_test(self, app_config: AppConfig) -> None:
        """测试步长小于测试期（窗口重叠）"""
        analyzer = WalkForwardAnalyzer(app_config)
        dates = pd.bdate_range(start="2024-01-01", periods=300)

        windows = analyzer._split_windows(
            dates, train_size=100, test_size=50, step_size=30
        )

        # 应该产生多个重叠窗口
        assert len(windows) >= 3

        # 验证窗口大小正确
        for train_dates, test_dates in windows:
            assert len(train_dates) == 100
            assert len(test_dates) == 50


# ============================================================
# Walk-Forward Run Tests
# ============================================================


class TestWalkForwardRun:
    """滚动前进分析运行测试"""

    def test_run_basic(self, app_config: AppConfig) -> None:
        """测试基本滚动前进分析流程"""
        import quant_trading.strategy  # noqa: F401

        analyzer = WalkForwardAnalyzer(app_config)
        data = make_synthetic_data(n_days=300, start_date="2024-01-01")

        result = analyzer.run(
            strategy_name="ma_crossover",
            param_grid={"short_window": [3, 5], "long_window": [15, 20]},
            data=data,
            symbol="000001",
            train_size=120,
            test_size=60,
            step_size=60,
            optimize_metric="sharpe_ratio",
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) >= 1
        assert isinstance(result.overall_performance, PerformanceResult)

    def test_run_result_windows_have_params(self, app_config: AppConfig) -> None:
        """测试每个窗口都有最优参数"""
        import quant_trading.strategy  # noqa: F401

        analyzer = WalkForwardAnalyzer(app_config)
        data = make_synthetic_data(n_days=300, start_date="2024-01-01")

        result = analyzer.run(
            strategy_name="ma_crossover",
            param_grid={"short_window": [3, 5], "long_window": [15, 20]},
            data=data,
            symbol="000001",
            train_size=120,
            test_size=60,
            step_size=60,
        )

        for window in result.windows:
            assert isinstance(window, WalkForwardWindow)
            assert "short_window" in window.best_params
            assert "long_window" in window.best_params
            assert isinstance(window.train_result, PerformanceResult)
            assert isinstance(window.test_result, PerformanceResult)

    def test_run_overall_equity_not_empty(self, app_config: AppConfig) -> None:
        """测试拼接后的整体净值曲线非空"""
        import quant_trading.strategy  # noqa: F401

        analyzer = WalkForwardAnalyzer(app_config)
        data = make_synthetic_data(n_days=300, start_date="2024-01-01")

        result = analyzer.run(
            strategy_name="ma_crossover",
            param_grid={"short_window": [3, 5], "long_window": [15, 20]},
            data=data,
            symbol="000001",
            train_size=120,
            test_size=60,
            step_size=60,
        )

        assert not result.overall_test_equity.empty
        assert len(result.overall_test_equity) > 0

    def test_run_data_too_short_raises(self, app_config: AppConfig) -> None:
        """测试数据不足时抛出异常"""
        import quant_trading.strategy  # noqa: F401

        analyzer = WalkForwardAnalyzer(app_config)
        data = make_synthetic_data(n_days=50, start_date="2024-01-01")

        with pytest.raises(ValueError, match="数据不足"):
            analyzer.run(
                strategy_name="ma_crossover",
                param_grid={"short_window": [3, 5], "long_window": [15, 20]},
                data=data,
                symbol="000001",
                train_size=120,
                test_size=60,
                step_size=60,
            )

    def test_run_with_total_return_metric(self, app_config: AppConfig) -> None:
        """测试使用 total_return 作为优化指标"""
        import quant_trading.strategy  # noqa: F401

        analyzer = WalkForwardAnalyzer(app_config)
        data = make_synthetic_data(n_days=300, start_date="2024-01-01")

        result = analyzer.run(
            strategy_name="ma_crossover",
            param_grid={"short_window": [3, 5], "long_window": [15, 20]},
            data=data,
            symbol="000001",
            train_size=120,
            test_size=60,
            step_size=60,
            optimize_metric="total_return",
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) >= 1


# ============================================================
# Equity Curve Stitching Tests
# ============================================================


class TestEquityCurveStitching:
    """净值曲线拼接测试"""

    def test_stitch_empty(self) -> None:
        """测试空列表"""
        result = WalkForwardAnalyzer._stitch_equity_curves([])
        assert result.empty

    def test_stitch_single(self) -> None:
        """测试单段曲线"""
        curve = pd.Series(
            [100.0, 105.0, 110.0],
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        result = WalkForwardAnalyzer._stitch_equity_curves([curve])
        assert len(result) == 3
        assert result.iloc[0] == 100.0
        assert result.iloc[-1] == 110.0

    def test_stitch_two_curves_continuity(self) -> None:
        """测试两段曲线拼接后连续性"""
        curve1 = pd.Series(
            [100.0, 105.0, 110.0],
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        curve2 = pd.Series(
            [200.0, 210.0, 220.0],
            index=pd.bdate_range("2024-01-04", periods=3),
        )

        result = WalkForwardAnalyzer._stitch_equity_curves([curve1, curve2])

        assert len(result) == 6
        # 第一段末尾值为 110
        assert result.iloc[2] == 110.0
        # 第二段应缩放使起始值 = 110
        assert abs(result.iloc[3] - 110.0) < 0.001
        # 第二段涨 10%: 110 * (220/200) = 121
        assert abs(result.iloc[-1] - 121.0) < 0.001


# ============================================================
# Summary Output Tests
# ============================================================


class TestWalkForwardSummary:
    """结果摘要输出测试"""

    def test_summary_basic(self, app_config: AppConfig) -> None:
        """测试 summary 方法输出格式"""
        import quant_trading.strategy  # noqa: F401

        analyzer = WalkForwardAnalyzer(app_config)
        data = make_synthetic_data(n_days=300, start_date="2024-01-01")

        result = analyzer.run(
            strategy_name="ma_crossover",
            param_grid={"short_window": [3, 5], "long_window": [15, 20]},
            data=data,
            symbol="000001",
            train_size=120,
            test_size=60,
            step_size=60,
        )

        summary = result.summary()
        assert "滚动前进分析报告" in summary
        assert "窗口数量" in summary
        assert "整体测试收益率" in summary
        assert "窗口明细" in summary

    def test_window_to_dict(self, app_config: AppConfig) -> None:
        """测试 WalkForwardWindow.to_dict"""
        import quant_trading.strategy  # noqa: F401

        analyzer = WalkForwardAnalyzer(app_config)
        data = make_synthetic_data(n_days=300, start_date="2024-01-01")

        result = analyzer.run(
            strategy_name="ma_crossover",
            param_grid={"short_window": [3, 5], "long_window": [15, 20]},
            data=data,
            symbol="000001",
            train_size=120,
            test_size=60,
            step_size=60,
        )

        for window in result.windows:
            d = window.to_dict()
            assert "train_start" in d
            assert "test_end" in d
            assert "best_params" in d
            assert "train_return" in d
            assert "test_return" in d
