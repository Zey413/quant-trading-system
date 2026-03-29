"""策略参数优化 - 简单网格搜索

对策略参数进行网格搜索，在给定参数空间中找到使指定指标最优的参数组合。

Example:
    >>> from quant_trading.backtest.optimizer import GridSearchOptimizer
    >>> optimizer = GridSearchOptimizer(engine)
    >>> param_grid = {
    ...     'short_window': [3, 5, 10],
    ...     'long_window': [15, 20, 30],
    ... }
    >>> result = optimizer.optimize('ma_crossover', param_grid, data, '000001')
    >>> print(result.best_params)
    >>> print(result.best_metric_value)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import pandas as pd

from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceResult
from quant_trading.core.config import AppConfig
from quant_trading.strategy.base import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """网格搜索优化结果

    Attributes:
        best_params: 最优参数组合
        best_metric_value: 最优指标值
        all_results: 所有参数组合及其对应指标的 DataFrame
        metric_name: 优化所使用的指标名称
    """

    best_params: dict[str, Any]
    best_metric_value: float
    all_results: pd.DataFrame
    metric_name: str = ""

    def summary(self) -> str:
        """格式化输出优化摘要"""
        lines = [
            "=" * 60,
            "              参数优化结果",
            "=" * 60,
            f"  优化指标:        {self.metric_name}",
            f"  最优指标值:      {self.best_metric_value:.6f}",
            f"  最优参数:        {self.best_params}",
            f"  测试组合数:      {len(self.all_results)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# 指标越大越好的列表（用于确定排序方向）
_HIGHER_IS_BETTER = {
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "win_rate",
    "profit_loss_ratio",
}


class GridSearchOptimizer:
    """网格搜索优化器

    对已注册策略的参数进行穷举搜索，为每组参数运行完整回测，
    并按指定指标选出最优参数组合。

    Args:
        config: 全局配置（用于构建 BacktestEngine）

    Example:
        >>> config = AppConfig()
        >>> optimizer = GridSearchOptimizer(config)
        >>> param_grid = {
        ...     'short_window': [3, 5, 10],
        ...     'long_window': [15, 20, 30],
        ... }
        >>> result = optimizer.optimize(
        ...     strategy_name='ma_crossover',
        ...     param_grid=param_grid,
        ...     data=data,
        ...     symbol='000001',
        ... )
        >>> print(result.best_params)
        >>> print(result.best_metric_value)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def optimize(
        self,
        strategy_name: str,
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        symbol: str = "unknown",
        metric: str = "sharpe_ratio",
    ) -> OptimizationResult:
        """执行网格搜索优化

        遍历 param_grid 中所有参数组合，为每组实例化策略并运行回测，
        最终按 metric 排序选出最优组合。

        Args:
            strategy_name: 已在 StrategyRegistry 注册的策略名称。
            param_grid: 参数网格，如 ``{'short_window': [3, 5], 'long_window': [15, 20]}``。
            data: 行情数据 DataFrame。
            symbol: 股票代码。
            metric: 用于排序的绩效指标，必须是 PerformanceResult 的属性名。
                    默认 ``'sharpe_ratio'``。

        Returns:
            OptimizationResult: 包含最优参数、最优指标值、所有结果的对象。

        Raises:
            KeyError: strategy_name 未注册。
            ValueError: param_grid 为空或 metric 不合法。
        """
        if not param_grid:
            raise ValueError("param_grid 不能为空")

        # 验证 metric
        valid_metrics = {
            f.name
            for f in PerformanceResult.__dataclass_fields__.values()
            if isinstance(f.default, (int, float))
            or f.type in ("float", "int")
        }
        # 动态获取有效指标名
        _sample = PerformanceResult.__dataclass_fields__
        numeric_fields = {
            name
            for name, fld in _sample.items()
            if fld.type in ("float", "int")
        }
        if metric not in numeric_fields:
            raise ValueError(
                f"无效的指标: {metric!r}。可用指标: {sorted(numeric_fields)}"
            )

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(
            "开始网格搜索: 策略=%s, 指标=%s, 共 %d 种组合",
            strategy_name,
            metric,
            len(combinations),
        )

        results_rows: list[dict[str, Any]] = []

        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                strategy = StrategyRegistry.get(strategy_name, **params)
            except (TypeError, ValueError) as e:
                logger.warning("跳过参数组合 %s: %s", params, e)
                continue

            # 每次优化都用新的 engine 实例，确保状态隔离
            engine = BacktestEngine(self.config)

            try:
                result = engine.run(strategy, data.copy(), symbol)
            except Exception as e:
                logger.warning("回测失败，参数 %s: %s", params, e)
                continue

            metric_value = getattr(result, metric, 0.0)

            row: dict[str, Any] = {**params, metric: metric_value}
            # 额外记录几个常用指标供参考
            for extra in (
                "total_return",
                "annualized_return",
                "max_drawdown",
                "sharpe_ratio",
                "total_trades",
            ):
                if extra != metric:
                    row[extra] = getattr(result, extra, 0.0)

            results_rows.append(row)

            logger.debug("参数 %s => %s=%.6f", params, metric, metric_value)

        if not results_rows:
            logger.warning("所有参数组合都失败，无有效结果")
            return OptimizationResult(
                best_params={},
                best_metric_value=0.0,
                all_results=pd.DataFrame(),
                metric_name=metric,
            )

        # 构建 DataFrame 并排序
        all_results_df = pd.DataFrame(results_rows)
        ascending = metric not in _HIGHER_IS_BETTER
        all_results_df = all_results_df.sort_values(
            by=metric, ascending=ascending
        ).reset_index(drop=True)

        # 提取最优
        best_row = all_results_df.iloc[0]
        best_params = {name: best_row[name] for name in param_names}
        best_metric_value = float(best_row[metric])

        logger.info(
            "网格搜索完成: 最优参数=%s, %s=%.6f",
            best_params,
            metric,
            best_metric_value,
        )

        return OptimizationResult(
            best_params=best_params,
            best_metric_value=best_metric_value,
            all_results=all_results_df,
            metric_name=metric,
        )
