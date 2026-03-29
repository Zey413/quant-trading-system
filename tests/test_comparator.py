"""策略比较器测试

测试覆盖：
- StrategyComparator: 添加策略、运行对比、同名策略处理
- ComparisonResult: 排名构建、摘要输出格式
- 边界情况: 空策略列表、策略失败跳过
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.comparator import ComparisonResult, StrategyComparator
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceMetrics, PerformanceResult
from quant_trading.core.config import (
    AppConfig,
    BacktestConfig,
    BrokerConfig,
    RiskConfig,
)


# ============================================================
# Fixtures & Helpers
# ============================================================


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


def make_synthetic_data(
    n_days: int = 120,
    start_price: float = 10.0,
    daily_return: float = 0.002,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """生成合成行情数据"""
    dates = pd.bdate_range(start=start_date, periods=n_days)
    np.random.seed(42)
    prices = [start_price]
    for i in range(1, n_days):
        # 添加一些随机波动使策略产生不同行为
        noise = np.random.normal(0, 0.005)
        prices.append(prices[-1] * (1 + daily_return + noise))

    return pd.DataFrame(
        {
            "date": dates,
            "open": [p * 0.998 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000 + np.random.randint(-100_000, 100_000)
                       for _ in range(n_days)],
        }
    )


# ============================================================
# StrategyComparator Tests
# ============================================================


class TestStrategyComparator:
    """策略比较器测试"""

    def test_compare_basic(self, app_config: AppConfig) -> None:
        """测试基本对比流程 - 2个策略"""
        # 导入策略模块以触发注册
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("rsi", period=14)

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001")

        assert isinstance(result, ComparisonResult)
        assert len(result.results) == 2
        assert "ma_crossover" in result.results
        assert "rsi" in result.results

    def test_compare_three_strategies(self, app_config: AppConfig) -> None:
        """测试3个策略对比"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("rsi", period=14)
        comparator.add_strategy("macd")

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001")

        assert len(result.results) == 3
        assert "macd" in result.results

    def test_compare_duplicate_strategy_names(self, app_config: AppConfig) -> None:
        """测试同名策略自动编号"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("ma_crossover", short_window=10, long_window=30)

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001")

        # 应有2个不同名称的结果
        assert len(result.results) == 2
        names = list(result.results.keys())
        assert names[0] != names[1]
        assert "ma_crossover" in names[0]
        assert "ma_crossover" in names[1]

    def test_compare_no_strategies_raises(self, app_config: AppConfig) -> None:
        """测试未添加策略时报错"""
        comparator = StrategyComparator(app_config)
        data = make_synthetic_data(n_days=60, start_date="2024-01-01")

        with pytest.raises(ValueError, match="未添加任何策略"):
            comparator.compare(data, symbol="000001")

    def test_compare_invalid_strategy_skipped(self, app_config: AppConfig) -> None:
        """测试无效策略被跳过"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        # 添加一个不存在的策略
        comparator.add_strategy("nonexistent_strategy")

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001")

        # 只有 ma_crossover 成功
        assert len(result.results) == 1
        assert "ma_crossover" in result.results

    def test_compare_rank_by_parameter(self, app_config: AppConfig) -> None:
        """测试按不同指标排名"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("rsi", period=14)

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001", rank_by="total_return")

        assert result.rank_metric == "total_return"
        assert "total_return" in result.ranking.columns


# ============================================================
# ComparisonResult Tests
# ============================================================


class TestComparisonResult:
    """比较结果测试"""

    def test_ranking_order(self, app_config: AppConfig) -> None:
        """测试排名按指标正确排序"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("rsi", period=14)
        comparator.add_strategy("macd")

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001", rank_by="sharpe_ratio")

        # 排名应按 sharpe_ratio 降序
        if len(result.ranking) >= 2:
            sharpe_values = result.ranking["sharpe_ratio"].tolist()
            assert sharpe_values == sorted(sharpe_values, reverse=True)

    def test_ranking_max_drawdown_ascending(self, app_config: AppConfig) -> None:
        """测试以最大回撤排名时升序排列 (越小越好)"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("rsi", period=14)

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001", rank_by="max_drawdown")

        # max_drawdown 不在 _HIGHER_IS_BETTER 中，应升序排列
        if len(result.ranking) >= 2:
            dd_values = result.ranking["max_drawdown"].tolist()
            assert dd_values == sorted(dd_values)

    def test_summary_output_format(self, app_config: AppConfig) -> None:
        """测试摘要输出格式"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("rsi", period=14)

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001")
        summary = result.summary()

        # 检查摘要包含关键内容
        assert "策略对比报告" in summary
        assert "ma_crossover" in summary
        assert "rsi" in summary
        assert "总收益率" in summary
        assert "夏普比率" in summary
        assert "排名指标" in summary

    def test_summary_empty_results(self) -> None:
        """测试空结果的摘要"""
        result = ComparisonResult(
            results={},
            ranking=pd.DataFrame(),
            rank_metric="sharpe_ratio",
        )
        summary = result.summary()
        assert "无比较结果" in summary

    def test_results_contain_performance_data(self, app_config: AppConfig) -> None:
        """测试结果中包含完整的绩效数据"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001")

        perf = result.results["ma_crossover"]
        assert isinstance(perf, PerformanceResult)
        assert not perf.equity_curve.empty
        assert hasattr(perf, "total_return")
        assert hasattr(perf, "sharpe_ratio")
        assert hasattr(perf, "max_drawdown")
        assert hasattr(perf, "win_rate")

    def test_ranking_columns(self, app_config: AppConfig) -> None:
        """测试排名 DataFrame 包含正确的列"""
        import quant_trading.strategy  # noqa: F401

        comparator = StrategyComparator(app_config)
        comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        comparator.add_strategy("rsi", period=14)

        data = make_synthetic_data(n_days=120, start_date="2024-01-01")
        result = comparator.compare(data, symbol="000001")

        assert "strategy" in result.ranking.columns
        assert "total_return" in result.ranking.columns
        assert "sharpe_ratio" in result.ranking.columns
        assert "max_drawdown" in result.ranking.columns
        assert "win_rate" in result.ranking.columns
        assert len(result.ranking) == 2
