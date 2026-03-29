"""优化器与新功能测试

测试覆盖：
- GridSearchOptimizer: 网格搜索、最优参数选择、边界情况
- Config 校验: 日期格式、佣金率范围、初始资金、跨字段校验
- PortfolioManager.record_history / history_df
- setup_logging 基本调用
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceResult
from quant_trading.backtest.optimizer import GridSearchOptimizer, OptimizationResult
from quant_trading.backtest.portfolio import PortfolioManager
from quant_trading.core.config import (
    AppConfig,
    BacktestConfig,
    BrokerConfig,
    RiskConfig,
)
from quant_trading.core.logging_config import setup_logging


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
    n_days: int = 60,
    start_price: float = 10.0,
    daily_return: float = 0.002,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """生成合成行情数据"""
    dates = pd.bdate_range(start=start_date, periods=n_days)
    prices = [start_price]
    for _ in range(1, n_days):
        prices.append(prices[-1] * (1 + daily_return))

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


# ============================================================
# GridSearchOptimizer Tests
# ============================================================


class TestGridSearchOptimizer:
    """网格搜索优化器测试"""

    def test_optimize_basic(self, app_config: AppConfig) -> None:
        """测试基本优化流程 - 使用 ma_crossover 策略"""
        # 导入策略模块以触发注册
        import quant_trading.strategy  # noqa: F401

        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(
            n_days=120,
            start_price=10.0,
            daily_return=0.003,
            start_date="2024-01-01",
        )

        param_grid = {
            "short_window": [3, 5],
            "long_window": [15, 20],
        }

        result = optimizer.optimize(
            strategy_name="ma_crossover",
            param_grid=param_grid,
            data=data,
            symbol="000001",
            metric="sharpe_ratio",
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params  # 非空
        assert "short_window" in result.best_params
        assert "long_window" in result.best_params
        # 应有 2*2 = 4 组参数组合结果
        assert len(result.all_results) == 4
        # metric_name 应被设置
        assert result.metric_name == "sharpe_ratio"

    def test_optimize_selects_best(self, app_config: AppConfig) -> None:
        """测试最优参数选择 - 结果按指标降序排列"""
        import quant_trading.strategy  # noqa: F401

        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(
            n_days=120,
            start_price=10.0,
            daily_return=0.003,
            start_date="2024-01-01",
        )

        param_grid = {
            "short_window": [3, 5, 10],
            "long_window": [15, 20, 30],
        }

        result = optimizer.optimize(
            strategy_name="ma_crossover",
            param_grid=param_grid,
            data=data,
            symbol="000001",
            metric="sharpe_ratio",
        )

        # 验证 all_results 按 sharpe_ratio 降序排列
        sharpe_values = result.all_results["sharpe_ratio"].tolist()
        assert sharpe_values == sorted(sharpe_values, reverse=True)

        # best_metric_value 应等于第一行的 sharpe_ratio
        assert abs(result.best_metric_value - sharpe_values[0]) < 1e-10

    def test_optimize_with_total_return_metric(self, app_config: AppConfig) -> None:
        """测试使用 total_return 作为优化指标"""
        import quant_trading.strategy  # noqa: F401

        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(
            n_days=120,
            start_price=10.0,
            daily_return=0.003,
            start_date="2024-01-01",
        )

        param_grid = {
            "short_window": [3, 5],
            "long_window": [15, 20],
        }

        result = optimizer.optimize(
            strategy_name="ma_crossover",
            param_grid=param_grid,
            data=data,
            symbol="000001",
            metric="total_return",
        )

        assert result.metric_name == "total_return"
        assert isinstance(result.best_metric_value, float)

    def test_optimize_invalid_strategy(self, app_config: AppConfig) -> None:
        """测试使用不存在的策略名"""
        import quant_trading.strategy  # noqa: F401

        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(n_days=60, start_date="2024-01-01")

        param_grid = {"short_window": [3, 5]}

        with pytest.raises(KeyError, match="未找到策略"):
            optimizer.optimize(
                strategy_name="nonexistent_strategy",
                param_grid=param_grid,
                data=data,
                symbol="000001",
            )

    def test_optimize_empty_param_grid(self, app_config: AppConfig) -> None:
        """测试空参数网格"""
        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(n_days=60, start_date="2024-01-01")

        with pytest.raises(ValueError, match="不能为空"):
            optimizer.optimize(
                strategy_name="ma_crossover",
                param_grid={},
                data=data,
                symbol="000001",
            )

    def test_optimize_invalid_metric(self, app_config: AppConfig) -> None:
        """测试无效的优化指标"""
        import quant_trading.strategy  # noqa: F401

        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(n_days=60, start_date="2024-01-01")

        param_grid = {"short_window": [3, 5], "long_window": [15, 20]}

        with pytest.raises(ValueError, match="无效的指标"):
            optimizer.optimize(
                strategy_name="ma_crossover",
                param_grid=param_grid,
                data=data,
                symbol="000001",
                metric="nonexistent_metric",
            )

    def test_optimize_skips_invalid_params(self, app_config: AppConfig) -> None:
        """测试跳过无效参数组合（如 short >= long）"""
        import quant_trading.strategy  # noqa: F401

        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(
            n_days=120,
            start_price=10.0,
            daily_return=0.003,
            start_date="2024-01-01",
        )

        # short_window=20, long_window=15 违反 short < long 约束
        param_grid = {
            "short_window": [5, 20],
            "long_window": [15, 30],
        }

        result = optimizer.optimize(
            strategy_name="ma_crossover",
            param_grid=param_grid,
            data=data,
            symbol="000001",
        )

        # 4种组合中 (20, 15) 无效，应被跳过 => 最多3条有效结果
        assert len(result.all_results) <= 3
        # 最优参数不应包含无效组合
        assert result.best_params.get("short_window", 0) < result.best_params.get(
            "long_window", float("inf")
        )

    def test_optimization_result_summary(self, app_config: AppConfig) -> None:
        """测试 OptimizationResult.summary() 输出"""
        import quant_trading.strategy  # noqa: F401

        optimizer = GridSearchOptimizer(app_config)
        data = make_synthetic_data(
            n_days=120,
            start_price=10.0,
            daily_return=0.003,
            start_date="2024-01-01",
        )

        param_grid = {
            "short_window": [3, 5],
            "long_window": [15, 20],
        }

        result = optimizer.optimize(
            strategy_name="ma_crossover",
            param_grid=param_grid,
            data=data,
            symbol="000001",
        )

        summary = result.summary()
        assert "参数优化结果" in summary
        assert "最优指标值" in summary
        assert "最优参数" in summary


# ============================================================
# Config Validation Tests
# ============================================================


class TestConfigValidation:
    """配置校验测试"""

    def test_valid_config(self) -> None:
        """测试合法配置"""
        config = AppConfig()
        errors = config.validate()
        assert errors == []

    def test_valid_date_yyyymmdd(self) -> None:
        """测试 YYYYMMDD 日期格式"""
        config = BacktestConfig(start_date="20230101", end_date="20251231")
        assert config.start_date == "20230101"

    def test_invalid_start_date_format(self) -> None:
        """测试无效的 start_date 格式"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            BacktestConfig(start_date="2023/01/01", end_date="2025-12-31")

    def test_invalid_end_date_format(self) -> None:
        """测试无效的 end_date 格式"""
        with pytest.raises(Exception):
            BacktestConfig(start_date="2023-01-01", end_date="Dec-31-2025")

    def test_invalid_date_value(self) -> None:
        """测试不合法日期（如2月30日）"""
        with pytest.raises(Exception):
            BacktestConfig(start_date="2024-02-30", end_date="2025-12-31")

    def test_start_after_end(self) -> None:
        """测试 start_date 晚于 end_date"""
        with pytest.raises(Exception):
            BacktestConfig(start_date="2025-12-31", end_date="2023-01-01")

    def test_commission_rate_too_high(self) -> None:
        """测试佣金率超出上限"""
        with pytest.raises(Exception):
            BrokerConfig(commission_rate=0.05)

    def test_commission_rate_negative(self) -> None:
        """测试佣金率为负"""
        with pytest.raises(Exception):
            BrokerConfig(commission_rate=-0.001)

    def test_commission_rate_boundary(self) -> None:
        """测试佣金率边界值"""
        # 0 和 0.01 应合法
        c1 = BrokerConfig(commission_rate=0.0)
        assert c1.commission_rate == 0.0
        c2 = BrokerConfig(commission_rate=0.01)
        assert c2.commission_rate == 0.01

    def test_initial_capital_zero(self) -> None:
        """测试初始资金为 0"""
        with pytest.raises(Exception):
            BacktestConfig(initial_capital=0.0)

    def test_initial_capital_negative(self) -> None:
        """测试初始资金为负"""
        with pytest.raises(Exception):
            BacktestConfig(initial_capital=-100_000.0)

    def test_validate_method_reports_errors(self) -> None:
        """测试 AppConfig.validate() 方法正常工作"""
        config = AppConfig()
        errors = config.validate()
        assert isinstance(errors, list)
        assert len(errors) == 0


# ============================================================
# Portfolio History Tests
# ============================================================


class TestPortfolioHistory:
    """组合历史记录测试"""

    def test_record_history_empty(self) -> None:
        """测试空历史"""
        pm = PortfolioManager(initial_capital=1_000_000.0)
        df = pm.history_df
        assert df.empty
        assert list(df.columns) == [
            "date",
            "cash",
            "market_value",
            "total_value",
            "return_pct",
        ]

    def test_record_history_basic(self) -> None:
        """测试记录历史 - 无持仓"""
        pm = PortfolioManager(initial_capital=1_000_000.0)
        pm.record_history(date(2024, 1, 1), {})
        pm.record_history(date(2024, 1, 2), {})

        df = pm.history_df
        assert len(df) == 2
        assert df.iloc[0]["cash"] == 1_000_000.0
        assert df.iloc[0]["total_value"] == 1_000_000.0
        assert df.iloc[0]["market_value"] == 0.0
        assert df.iloc[0]["return_pct"] == 0.0

    def test_record_history_with_position(self) -> None:
        """测试记录历史 - 有持仓"""
        from quant_trading.core.enums import OrderSide, OrderStatus
        from quant_trading.core.models import Order

        pm = PortfolioManager(initial_capital=1_000_000.0)

        # 买入 1000 股 @ 10.0
        order = Order(
            order_id="test-001",
            date=date(2024, 1, 1),
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
            filled_price=10.0,
            filled_quantity=1000,
            commission=5.0,
            tax=0.0,
            status=OrderStatus.FILLED,
        )
        pm.process_buy(order)

        # 记录第一天（价格10.0）
        pm.record_history(date(2024, 1, 1), {"000001": 10.0})

        # 记录第二天（价格涨到11.0）
        pm.record_history(date(2024, 1, 2), {"000001": 11.0})

        df = pm.history_df
        assert len(df) == 2

        # 第一天: cash = 1M - 10*1000 - 5 = 989995, market_value = 10000
        assert df.iloc[0]["market_value"] == 10_000.0

        # 第二天: 价格涨到11, market_value = 11000
        assert df.iloc[1]["market_value"] == 11_000.0
        assert df.iloc[1]["total_value"] > df.iloc[0]["total_value"]


# ============================================================
# Logging Config Tests
# ============================================================


class TestLoggingConfig:
    """日志配置测试"""

    def test_setup_logging_default(self) -> None:
        """测试默认日志配置"""
        setup_logging(level="WARNING")
        import logging

        logger = logging.getLogger("quant_trading")
        assert logger.level == logging.WARNING
        assert len(logger.handlers) >= 1

    def test_setup_logging_debug(self) -> None:
        """测试 DEBUG 级别"""
        setup_logging(level="DEBUG")
        import logging

        logger = logging.getLogger("quant_trading")
        assert logger.level == logging.DEBUG

    def test_setup_logging_invalid_level(self) -> None:
        """测试无效日志级别"""
        with pytest.raises(ValueError, match="无效的日志级别"):
            setup_logging(level="INVALID")

    def test_setup_logging_with_file(self, tmp_path) -> None:
        """测试带文件的日志配置"""
        log_file = tmp_path / "test.log"
        setup_logging(level="INFO", log_file=str(log_file))

        import logging

        logger = logging.getLogger("quant_trading")
        logger.info("测试日志消息")

        # 验证文件被创建且写入
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "测试日志消息" in content

    def test_setup_logging_idempotent(self) -> None:
        """测试多次调用 setup_logging 不会产生重复 handler"""
        setup_logging(level="INFO")
        setup_logging(level="INFO")
        setup_logging(level="INFO")

        import logging

        logger = logging.getLogger("quant_trading")
        # 每次调用都会清理旧 handler，所以应只有1个
        assert len(logger.handlers) == 1
