"""滚动前进分析 (Walk-Forward Analysis)

将数据分成多个训练/测试窗口，在每个训练期优化参数，
在测试期验证效果，避免过拟合。

典型用法:
    >>> from quant_trading.backtest.walk_forward import WalkForwardAnalyzer
    >>> analyzer = WalkForwardAnalyzer(config)
    >>> result = analyzer.run(
    ...     strategy_name="ma_crossover",
    ...     param_grid={"short_window": [3, 5, 10], "long_window": [15, 20, 30]},
    ...     data=df,
    ...     symbol="000001",
    ...     train_size=120,
    ...     test_size=60,
    ...     step_size=60,
    ... )
    >>> print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceMetrics, PerformanceResult
from quant_trading.backtest.optimizer import GridSearchOptimizer, _HIGHER_IS_BETTER
from quant_trading.core.config import AppConfig, BacktestConfig
from quant_trading.strategy.base import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """单个滚动窗口的结果

    Attributes:
        train_start: 训练期开始日期
        train_end: 训练期结束日期
        test_start: 测试期开始日期
        test_end: 测试期结束日期
        best_params: 训练期优化出的最优参数
        train_result: 训练期回测绩效
        test_result: 测试期回测绩效
    """

    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict[str, Any]
    train_result: PerformanceResult
    test_result: PerformanceResult

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "best_params": self.best_params,
            "train_return": self.train_result.total_return,
            "test_return": self.test_result.total_return,
            "train_sharpe": self.train_result.sharpe_ratio,
            "test_sharpe": self.test_result.sharpe_ratio,
        }


@dataclass
class WalkForwardResult:
    """滚动前进分析结果

    Attributes:
        windows: 每个窗口的详细结果
        overall_test_equity: 拼接所有测试期的净值曲线
        overall_performance: 基于拼接净值曲线计算的整体绩效
    """

    windows: list[WalkForwardWindow]
    overall_test_equity: pd.Series
    overall_performance: PerformanceResult

    def summary(self) -> str:
        """格式化输出滚动前进分析摘要"""
        lines = [
            "=" * 60,
            "          滚动前进分析报告 (Walk-Forward)",
            "=" * 60,
            "",
            f"  窗口数量:            {len(self.windows)}",
            f"  整体测试收益率:      {self.overall_performance.total_return:>10.2%}",
            f"  整体测试夏普比率:    {self.overall_performance.sharpe_ratio:>10.4f}",
            f"  整体测试最大回撤:    {self.overall_performance.max_drawdown:>10.2%}",
            "",
            "-" * 60,
            "  窗口明细:",
            "-" * 60,
        ]

        for i, w in enumerate(self.windows, 1):
            lines.extend(
                [
                    f"  [{i}] 训练: {w.train_start} ~ {w.train_end}",
                    f"      测试: {w.test_start} ~ {w.test_end}",
                    f"      最优参数: {w.best_params}",
                    f"      训练收益: {w.train_result.total_return:>8.2%}  "
                    f"测试收益: {w.test_result.total_return:>8.2%}",
                    "",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class WalkForwardAnalyzer:
    """滚动前进分析器

    将数据按时间切割为多个「训练 + 测试」窗口，在每个训练期使用
    GridSearchOptimizer 寻找最优参数，然后在测试期验证策略表现。
    最后将所有测试期净值曲线拼接，计算整体绩效。

    Args:
        config: 全局配置

    Example:
        >>> analyzer = WalkForwardAnalyzer(config)
        >>> result = analyzer.run(
        ...     strategy_name="ma_crossover",
        ...     param_grid={"short_window": [3, 5, 10], "long_window": [15, 20, 30]},
        ...     data=df,
        ...     symbol="000001",
        ...     train_size=120,
        ...     test_size=60,
        ...     step_size=60,
        ... )
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _split_windows(
        self,
        dates: pd.DatetimeIndex,
        train_size: int,
        test_size: int,
        step_size: int,
    ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """将日期序列切分为(训练期, 测试期)窗口列表

        Args:
            dates: 排序后的日期索引
            train_size: 训练期交易日数
            test_size: 测试期交易日数
            step_size: 每次前进的交易日数

        Returns:
            [(train_dates, test_dates), ...] 列表
        """
        windows: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        total = len(dates)

        start = 0
        while start + train_size + test_size <= total:
            train_dates = dates[start : start + train_size]
            test_dates = dates[start + train_size : start + train_size + test_size]
            windows.append((train_dates, test_dates))
            start += step_size

        return windows

    def run(
        self,
        strategy_name: str,
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        symbol: str = "unknown",
        train_size: int = 120,
        test_size: int = 60,
        step_size: int = 60,
        optimize_metric: str = "sharpe_ratio",
    ) -> WalkForwardResult:
        """执行滚动前进分析

        Args:
            strategy_name: 已在 StrategyRegistry 注册的策略名称
            param_grid: 参数网格
            data: 行情数据 DataFrame
            symbol: 股票代码
            train_size: 训练期交易日数
            test_size: 测试期交易日数
            step_size: 每次前进交易日数
            optimize_metric: 优化指标名称

        Returns:
            WalkForwardResult: 滚动前进分析结果

        Raises:
            ValueError: 数据不足以构成至少一个窗口
        """
        # 准备数据：确保以日期为索引并排序
        df = data.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        dates = df.index

        # 切分窗口
        windows = self._split_windows(dates, train_size, test_size, step_size)

        if not windows:
            raise ValueError(
                f"数据不足: 共 {len(dates)} 个交易日，"
                f"需要至少 {train_size + test_size} 个交易日 "
                f"(train_size={train_size} + test_size={test_size})"
            )

        logger.info(
            "开始滚动前进分析: 策略=%s, 窗口数=%d, "
            "训练=%d天, 测试=%d天, 步长=%d天",
            strategy_name,
            len(windows),
            train_size,
            test_size,
            step_size,
        )

        window_results: list[WalkForwardWindow] = []
        test_equity_pieces: list[pd.Series] = []

        for i, (train_dates, test_dates) in enumerate(windows):
            train_start = str(train_dates[0].date())
            train_end = str(train_dates[-1].date())
            test_start = str(test_dates[0].date())
            test_end = str(test_dates[-1].date())

            logger.info(
                "窗口 [%d/%d]: 训练 %s~%s, 测试 %s~%s",
                i + 1,
                len(windows),
                train_start,
                train_end,
                test_start,
                test_end,
            )

            # --- 1. 在训练期优化参数 ---
            train_data = df.loc[train_dates[0] : train_dates[-1]].copy()
            # 需要重新加 date 列给 engine（engine 内部会处理）
            train_config = self.config.model_copy(deep=True)
            train_config.backtest = train_config.backtest.model_copy(
                update={
                    "start_date": train_start,
                    "end_date": train_end,
                }
            )

            optimizer = GridSearchOptimizer(train_config)
            opt_result = optimizer.optimize(
                strategy_name=strategy_name,
                param_grid=param_grid,
                data=train_data.reset_index().rename(columns={train_data.index.name or "index": "date"}),
                symbol=symbol,
                metric=optimize_metric,
            )

            best_params = opt_result.best_params

            if not best_params:
                logger.warning("窗口 [%d] 优化失败，跳过", i + 1)
                continue

            # 将 numpy 类型转为 Python 原生类型（DataFrame 提取的值可能是 np.float64）
            best_params = {
                k: int(v) if isinstance(v, (int, float, np.integer, np.floating)) and float(v) == int(v) else
                float(v) if isinstance(v, (np.floating,)) else v
                for k, v in best_params.items()
            }

            # --- 2. 用最优参数在训练期获取绩效（已在优化过程中计算） ---
            train_engine = BacktestEngine(train_config)
            train_strategy = StrategyRegistry.get(strategy_name, **best_params)
            train_result = train_engine.run(
                train_strategy,
                train_data.reset_index().rename(columns={train_data.index.name or "index": "date"}),
                symbol,
            )

            # --- 3. 在测试期用相同参数回测 ---
            test_data = df.loc[test_dates[0] : test_dates[-1]].copy()
            test_config = self.config.model_copy(deep=True)
            test_config.backtest = test_config.backtest.model_copy(
                update={
                    "start_date": test_start,
                    "end_date": test_end,
                }
            )

            test_engine = BacktestEngine(test_config)
            test_strategy = StrategyRegistry.get(strategy_name, **best_params)
            test_result = test_engine.run(
                test_strategy,
                test_data.reset_index().rename(columns={test_data.index.name or "index": "date"}),
                symbol,
            )

            logger.info(
                "窗口 [%d] 完成: 最优参数=%s, 训练收益=%.2f%%, 测试收益=%.2f%%",
                i + 1,
                best_params,
                train_result.total_return * 100,
                test_result.total_return * 100,
            )

            window_results.append(
                WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_params=best_params,
                    train_result=train_result,
                    test_result=test_result,
                )
            )

            # 收集测试期净值曲线
            if not test_result.equity_curve.empty:
                test_equity_pieces.append(test_result.equity_curve)

        # --- 4. 拼接所有测试期净值曲线 ---
        overall_test_equity = self._stitch_equity_curves(test_equity_pieces)

        # --- 5. 计算整体测试绩效 ---
        if overall_test_equity.empty:
            overall_performance = PerformanceMetrics._empty_result()
        else:
            overall_performance = PerformanceMetrics.calculate(
                equity_curve=overall_test_equity,
                trades=[],
                config=self.config.backtest,
            )

        logger.info(
            "滚动前进分析完成: %d 个窗口, 整体测试收益=%.2f%%",
            len(window_results),
            overall_performance.total_return * 100,
        )

        return WalkForwardResult(
            windows=window_results,
            overall_test_equity=overall_test_equity,
            overall_performance=overall_performance,
        )

    @staticmethod
    def _stitch_equity_curves(pieces: list[pd.Series]) -> pd.Series:
        """将多段测试期净值曲线拼接为连续曲线

        每段新曲线的起始值会被缩放，使其与前一段的结束值衔接。
        即：后一段的返回率乘以前一段末尾的净值。

        Args:
            pieces: 各测试窗口的净值曲线列表

        Returns:
            拼接后的连续净值曲线
        """
        if not pieces:
            return pd.Series(dtype=float)

        if len(pieces) == 1:
            return pieces[0].copy()

        stitched_parts: list[pd.Series] = [pieces[0].copy()]
        last_value = pieces[0].iloc[-1]

        for piece in pieces[1:]:
            if piece.empty:
                continue
            # 计算缩放因子：让此段起始值等于上一段末尾值
            scale = last_value / piece.iloc[0]
            scaled = piece * scale
            stitched_parts.append(scaled)
            last_value = scaled.iloc[-1]

        return pd.concat(stitched_parts)
