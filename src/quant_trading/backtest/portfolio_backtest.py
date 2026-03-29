"""多股票组合回测 - 同时回测多只股票的策略表现

支持等权重和自定义权重的多股票组合回测，汇总每只股票的单独绩效
并计算组合层面的整体绩效和股票间相关性。

典型用法:
    >>> from quant_trading.backtest.portfolio_backtest import PortfolioBacktester
    >>> backtester = PortfolioBacktester(config)
    >>> result = backtester.run(
    ...     strategy=strategy,
    ...     data_dict={"000001": df1, "600519": df2, "000858": df3},
    ...     weights="equal",
    ... )
    >>> print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pandas as pd

from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceMetrics, PerformanceResult
from quant_trading.core.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class PortfolioBacktestResult:
    """多股票组合回测结果

    Attributes:
        overall: 组合整体绩效
        per_stock: 每只股票的单独绩效 {symbol: PerformanceResult}
        stock_weights: 各股票权重 {symbol: weight}
        correlation_matrix: 股票日收益率之间的相关性矩阵
    """

    overall: PerformanceResult
    per_stock: dict[str, PerformanceResult]
    stock_weights: dict[str, float]
    correlation_matrix: pd.DataFrame

    def summary(self) -> str:
        """格式化输出多股票组合回测摘要"""
        lines = [
            "=" * 60,
            "           多股票组合回测报告",
            "=" * 60,
            "",
            "【组合整体绩效】",
            f"  总收益率:          {self.overall.total_return:>10.2%}",
            f"  年化收益率:        {self.overall.annualized_return:>10.2%}",
            f"  最大回撤:          {self.overall.max_drawdown:>10.2%}",
            f"  夏普比率:          {self.overall.sharpe_ratio:>10.4f}",
            "",
            "【个股权重】",
        ]

        for symbol, weight in sorted(self.stock_weights.items()):
            lines.append(f"  {symbol}: {weight:>8.2%}")

        lines.extend(
            [
                "",
                "【个股绩效】",
                "-" * 60,
            ]
        )

        for symbol in sorted(self.per_stock.keys()):
            result = self.per_stock[symbol]
            lines.extend(
                [
                    f"  {symbol}:",
                    f"    收益率: {result.total_return:>8.2%}  "
                    f"夏普: {result.sharpe_ratio:>8.4f}  "
                    f"回撤: {result.max_drawdown:>8.2%}",
                ]
            )

        lines.extend(
            [
                "",
                "【相关性矩阵】",
            ]
        )

        if not self.correlation_matrix.empty:
            lines.append(self.correlation_matrix.to_string())
        else:
            lines.append("  (数据不足，无法计算相关性)")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)


class PortfolioBacktester:
    """多股票组合回测器

    对每只股票独立运行回测，然后按权重加总净值曲线，
    计算组合层面的绩效指标和股票间的相关性。

    Args:
        config: 全局配置

    Example:
        >>> backtester = PortfolioBacktester(config)
        >>> result = backtester.run(
        ...     strategy=strategy,
        ...     data_dict={"000001": df1, "600519": df2},
        ...     weights="equal",
        ... )
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(
        self,
        strategy,
        data_dict: dict[str, pd.DataFrame],
        weights: Union[str, dict[str, float]] = "equal",
    ) -> PortfolioBacktestResult:
        """执行多股票组合回测

        对每只股票独立执行单标的回测，然后按权重合成组合净值曲线。

        Args:
            strategy: 策略实例或策略工厂（callable）。
                      如果是 callable，每只股票调用一次以获取独立实例。
                      如果是策略实例，每只股票会使用相同策略的 generate_signals。
            data_dict: {symbol: DataFrame} 多标的行情数据
            weights: 权重方案:
                - "equal": 等权重
                - dict: 自定义权重, 如 {"000001": 0.4, "600519": 0.6}

        Returns:
            PortfolioBacktestResult: 组合回测结果

        Raises:
            ValueError: data_dict 为空或权重不合法
        """
        if not data_dict:
            raise ValueError("data_dict 不能为空")

        symbols = list(data_dict.keys())

        # 解析权重
        stock_weights = self._resolve_weights(symbols, weights)

        logger.info(
            "开始多股票组合回测: %d 只股票, 权重=%s",
            len(symbols),
            stock_weights,
        )

        # --- 1. 对每只股票单独回测 ---
        per_stock: dict[str, PerformanceResult] = {}
        equity_curves: dict[str, pd.Series] = {}

        for symbol in symbols:
            data = data_dict[symbol]

            # 如果 strategy 是 callable (工厂函数)，每只股票创建新实例
            if callable(strategy) and not hasattr(strategy, "generate_signals"):
                strat = strategy()
            else:
                # 需要复制策略以避免信号列冲突（generate_signals 会修改 data）
                # 由于 engine.run 内部会 copy data, 所以直接使用即可
                strat = strategy

            engine = BacktestEngine(self.config)
            result = engine.run(strat, data.copy(), symbol)

            per_stock[symbol] = result
            if not result.equity_curve.empty:
                equity_curves[symbol] = result.equity_curve

            logger.info(
                "  %s: 收益率=%.2f%%, 夏普=%.4f, 回撤=%.2f%%",
                symbol,
                result.total_return * 100,
                result.sharpe_ratio,
                result.max_drawdown * 100,
            )

        # --- 2. 计算组合加权净值曲线 ---
        overall_equity = self._calc_portfolio_equity(
            equity_curves, stock_weights
        )

        # --- 3. 计算组合整体绩效 ---
        if overall_equity.empty:
            overall = PerformanceMetrics._empty_result()
        else:
            overall = PerformanceMetrics.calculate(
                equity_curve=overall_equity,
                trades=[],
                config=self.config.backtest,
            )

        # --- 4. 计算股票间相关性矩阵 ---
        correlation_matrix = self._calc_correlation_matrix(equity_curves)

        logger.info(
            "多股票组合回测完成: 组合收益率=%.2f%%, 夏普=%.4f",
            overall.total_return * 100,
            overall.sharpe_ratio,
        )

        return PortfolioBacktestResult(
            overall=overall,
            per_stock=per_stock,
            stock_weights=stock_weights,
            correlation_matrix=correlation_matrix,
        )

    @staticmethod
    def _resolve_weights(
        symbols: list[str],
        weights: Union[str, dict[str, float]],
    ) -> dict[str, float]:
        """解析权重配置

        Args:
            symbols: 股票代码列表
            weights: "equal" 或自定义权重字典

        Returns:
            归一化后的权重字典

        Raises:
            ValueError: 权重无效
        """
        if weights == "equal":
            n = len(symbols)
            return {s: 1.0 / n for s in symbols}

        if not isinstance(weights, dict):
            raise ValueError(
                f"weights 必须为 'equal' 或 dict, 收到 {type(weights)}"
            )

        # 验证所有股票都有权重
        missing = set(symbols) - set(weights.keys())
        if missing:
            raise ValueError(f"以下股票缺少权重: {missing}")

        # 归一化权重
        total = sum(weights[s] for s in symbols)
        if total <= 0:
            raise ValueError(f"权重之和必须大于 0, 当前为 {total}")

        return {s: weights[s] / total for s in symbols}

    def _calc_portfolio_equity(
        self,
        equity_curves: dict[str, pd.Series],
        weights: dict[str, float],
    ) -> pd.Series:
        """按权重加总各股票净值曲线，得到组合净值曲线

        将每只股票的净值曲线归一化为收益率序列，按权重加总后
        还原为以初始资金为基准的组合净值。

        Args:
            equity_curves: {symbol: equity_series}
            weights: {symbol: weight}

        Returns:
            组合净值曲线
        """
        if not equity_curves:
            return pd.Series(dtype=float)

        initial_capital = self.config.backtest.initial_capital

        # 将每只股票的净值转为日收益率
        returns_dict: dict[str, pd.Series] = {}
        for symbol, curve in equity_curves.items():
            if curve.empty or len(curve) < 2:
                continue
            daily_ret = curve.pct_change().fillna(0.0)
            returns_dict[symbol] = daily_ret

        if not returns_dict:
            return pd.Series(dtype=float)

        # 对齐日期
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(0.0)

        # 按权重加总
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for symbol in returns_dict:
            w = weights.get(symbol, 0.0)
            portfolio_returns += returns_df[symbol] * w

        # 还原为净值曲线
        portfolio_equity = initial_capital * (1 + portfolio_returns).cumprod()

        return portfolio_equity

    @staticmethod
    def _calc_correlation_matrix(
        equity_curves: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """计算各股票日收益率之间的 Pearson 相关性矩阵

        Args:
            equity_curves: {symbol: equity_series}

        Returns:
            相关性矩阵 DataFrame
        """
        if len(equity_curves) < 2:
            return pd.DataFrame()

        returns_dict: dict[str, pd.Series] = {}
        for symbol, curve in equity_curves.items():
            if curve.empty or len(curve) < 2:
                continue
            returns_dict[symbol] = curve.pct_change().dropna()

        if len(returns_dict) < 2:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_dict).dropna()

        if returns_df.empty or len(returns_df) < 2:
            return pd.DataFrame()

        return returns_df.corr()
