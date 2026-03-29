"""策略比较器 - 对比多个策略的回测表现

在相同数据和配置下运行多个策略，并对比结果，
支持按指定指标排名、格式化摘要、以及多净值曲线对比图。

Example:
    >>> from quant_trading.backtest.comparator import StrategyComparator
    >>> comparator = StrategyComparator(config)
    >>> comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
    >>> comparator.add_strategy("rsi", period=14)
    >>> comparator.add_strategy("macd")
    >>> result = comparator.compare(data, symbol="000001")
    >>> print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceResult
from quant_trading.core.config import AppConfig
from quant_trading.strategy.base import StrategyRegistry

logger = logging.getLogger(__name__)

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

# 用于摘要表格的核心指标
_SUMMARY_METRICS = [
    ("total_return", "总收益率", True),
    ("annualized_return", "年化收益率", True),
    ("max_drawdown", "最大回撤", False),
    ("sharpe_ratio", "夏普比率", False),
    ("sortino_ratio", "Sortino比率", False),
    ("calmar_ratio", "Calmar比率", False),
    ("total_trades", "交易次数", False),
    ("win_rate", "胜率", True),
    ("profit_loss_ratio", "盈亏比", False),
]


@dataclass
class ComparisonResult:
    """策略比较结果

    Attributes:
        results: {strategy_name: PerformanceResult} 各策略的回测结果
        ranking: 按指标排名的 DataFrame
        rank_metric: 排名所用指标名
    """

    results: dict[str, PerformanceResult]
    ranking: pd.DataFrame
    rank_metric: str = "sharpe_ratio"

    def summary(self) -> str:
        """格式化对比摘要

        生成多策略关键指标并排对比表格。

        Returns:
            格式化的字符串
        """
        if not self.results:
            return "无比较结果"

        names = list(self.results.keys())
        col_width = max(14, max(len(n) for n in names) + 2)

        lines: list[str] = []
        lines.append("=" * (16 + col_width * len(names)))
        lines.append("                  策略对比报告")
        lines.append("=" * (16 + col_width * len(names)))
        lines.append("")

        # 表头
        header = f"{'指标':<14}"
        for name in names:
            header += f"  {name:>{col_width - 2}}"
        lines.append(header)

        sep = f"{'-' * 14}"
        for _ in names:
            sep += f"  {'-' * (col_width - 2)}"
        lines.append(sep)

        # 每个指标一行
        for metric_key, metric_label, is_pct in _SUMMARY_METRICS:
            row = f"{metric_label:<14}"
            for name in names:
                result = self.results[name]
                value = getattr(result, metric_key, 0.0)
                if is_pct:
                    cell = f"{value:>{col_width - 2}.2%}"
                elif metric_key == "total_trades":
                    cell = f"{int(value):>{col_width - 2}d}"
                else:
                    cell = f"{value:>{col_width - 2}.4f}"
                row += f"  {cell}"
            lines.append(row)

        lines.append("")

        # 排名信息
        lines.append(f"排名指标: {self.rank_metric}")
        if not self.ranking.empty:
            lines.append("排名:")
            for i, row in self.ranking.iterrows():
                metric_val = row.get(self.rank_metric, 0.0)
                lines.append(
                    f"  {i + 1}. {row['strategy']:<20} "
                    f"{self.rank_metric}={metric_val:.4f}"
                )

        lines.append("")
        lines.append("=" * (16 + col_width * len(names)))
        return "\n".join(lines)


class StrategyComparator:
    """策略比较器

    在相同数据和配置下运行多个策略，并对比结果。

    Args:
        config: 全局配置（用于构建 BacktestEngine）

    Example:
        >>> config = AppConfig()
        >>> comparator = StrategyComparator(config)
        >>> comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
        >>> comparator.add_strategy("rsi", period=14)
        >>> comparator.add_strategy("macd")
        >>> result = comparator.compare(data, symbol="000001")
        >>> print(result.summary())
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._strategies: list[tuple[str, str, dict[str, Any]]] = []

    def add_strategy(self, name: str, **kwargs: Any) -> None:
        """添加待比较的策略

        通过 StrategyRegistry 名称和参数添加策略。如果同一策略名添加
        多次（参数不同），会自动添加编号后缀以区分。

        Args:
            name: 策略注册名（如 'ma_crossover', 'rsi', 'macd'）
            **kwargs: 传递给策略构造函数的参数
        """
        # 生成显示名称（处理同名策略）
        display_name = name
        existing_names = {s[0] for s in self._strategies}
        if display_name in existing_names:
            suffix = 2
            while f"{name}_{suffix}" in existing_names:
                suffix += 1
            display_name = f"{name}_{suffix}"

        self._strategies.append((display_name, name, kwargs))
        logger.info("添加策略: %s (注册名=%s, 参数=%s)", display_name, name, kwargs)

    def compare(
        self,
        data: pd.DataFrame,
        symbol: str = "unknown",
        rank_by: str = "sharpe_ratio",
    ) -> ComparisonResult:
        """运行所有策略并比较

        为每个策略创建独立的 BacktestEngine，在相同数据上运行回测，
        并按 rank_by 指标排名。

        Args:
            data: 行情数据 DataFrame（标准 OHLCV 格式）
            symbol: 股票代码
            rank_by: 用于排名的指标，默认 'sharpe_ratio'

        Returns:
            ComparisonResult: 包含所有策略结果和排名

        Raises:
            ValueError: 未添加任何策略
        """
        if not self._strategies:
            raise ValueError("未添加任何策略，请先调用 add_strategy()")

        results: dict[str, PerformanceResult] = {}

        logger.info(
            "开始策略对比: %d 个策略, 标的=%s",
            len(self._strategies),
            symbol,
        )

        for display_name, registry_name, kwargs in self._strategies:
            try:
                strategy = StrategyRegistry.get(registry_name, **kwargs)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("跳过策略 %s: %s", display_name, e)
                continue

            engine = BacktestEngine(self.config)

            try:
                result = engine.run(strategy, data.copy(), symbol)
                results[display_name] = result
                logger.info(
                    "策略 %s 完成: 总收益率=%.2f%%, 夏普=%.4f",
                    display_name,
                    result.total_return * 100,
                    result.sharpe_ratio,
                )
            except Exception as e:
                logger.warning("策略 %s 回测失败: %s", display_name, e)
                continue

        # 构建排名 DataFrame
        ranking = self._build_ranking(results, rank_by)

        return ComparisonResult(
            results=results,
            ranking=ranking,
            rank_metric=rank_by,
        )

    def _build_ranking(
        self,
        results: dict[str, PerformanceResult],
        rank_by: str,
    ) -> pd.DataFrame:
        """构建排名 DataFrame

        Args:
            results: 策略结果字典
            rank_by: 排名指标

        Returns:
            按指标排名的 DataFrame
        """
        if not results:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for name, result in results.items():
            row: dict[str, Any] = {"strategy": name}
            for metric_key, _, _ in _SUMMARY_METRICS:
                row[metric_key] = getattr(result, metric_key, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows)
        ascending = rank_by not in _HIGHER_IS_BETTER
        df = df.sort_values(by=rank_by, ascending=ascending).reset_index(drop=True)

        return df

    def plot_comparison(
        self,
        result: ComparisonResult,
        save_path: Optional[str] = None,
    ) -> None:
        """绘制策略对比图 (多条净值曲线)

        将所有策略的净值曲线归一化后绘制在同一张图上，
        便于直观比较策略表现。

        Args:
            result: 策略比较结果
            save_path: 图片保存路径，为 None 时直接显示
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 中文支持
        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei",
            "Arial Unicode MS",
            "WenQuanYi Micro Hei",
            "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

        if not result.results:
            logger.warning("无策略结果，无法绘图")
            return

        fig, (ax_equity, ax_dd) = plt.subplots(
            2,
            1,
            figsize=(14, 9),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.suptitle("策略净值对比", fontsize=16, fontweight="bold", y=0.95)

        # 配色方案
        colors = [
            "#1976D2",
            "#EF5350",
            "#4CAF50",
            "#FF9800",
            "#9C27B0",
            "#00BCD4",
            "#795548",
            "#607D8B",
        ]

        for idx, (name, perf_result) in enumerate(result.results.items()):
            equity = perf_result.equity_curve
            if equity.empty:
                continue

            # 归一化到起始值为1
            normalized = equity / equity.iloc[0]
            if not isinstance(normalized.index, pd.DatetimeIndex):
                normalized.index = pd.to_datetime(normalized.index)

            color = colors[idx % len(colors)]

            # 净值曲线
            ax_equity.plot(
                normalized.index,
                normalized.values,
                color=color,
                linewidth=1.5,
                label=name,
            )

            # 回撤曲线
            cummax = normalized.cummax()
            drawdown = (normalized - cummax) / cummax
            ax_dd.plot(
                drawdown.index,
                drawdown.values,
                color=color,
                linewidth=0.8,
                label=name,
            )

        # 美化
        ax_equity.set_ylabel("归一化净值", fontsize=11)
        ax_equity.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax_equity.grid(True, alpha=0.3, color="#E0E0E0")
        ax_equity.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

        import matplotlib.ticker as mticker

        ax_dd.set_ylabel("回撤", fontsize=10)
        ax_dd.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1.0, decimals=0)
        )
        ax_dd.grid(True, alpha=0.3, color="#E0E0E0")
        ax_dd.legend(loc="lower left", fontsize=8, framealpha=0.9)

        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_dd.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=45)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            logger.info("策略对比图已保存至: %s", save_path)
            plt.close(fig)
        else:
            plt.show()
