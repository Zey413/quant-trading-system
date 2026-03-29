"""因子分析器 - IC/IR 分析与分层回测

提供:
- FactorAnalyzer: 因子分析器
  - IC (Information Coefficient) 分析
  - IR (Information Ratio) 分析
  - 分层回测 (Quantile Backtest)
  - 因子收益率分析
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from quant_trading.backtest.metrics import TRADING_DAYS_PER_YEAR
from quant_trading.factor.base import Factor

logger = logging.getLogger(__name__)


@dataclass
class FactorAnalysisResult:
    """因子分析结果"""

    factor_name: str

    # IC 分析
    ic_series: pd.Series            # 每期 IC 值
    ic_mean: float                  # IC 均值
    ic_std: float                   # IC 标准差
    ic_ir: float                    # IC_IR = IC均值 / IC标准差
    ic_positive_ratio: float        # IC > 0 的比例
    rank_ic_series: pd.Series       # 每期 Rank IC 值
    rank_ic_mean: float             # Rank IC 均值
    rank_ic_ir: float               # Rank IC IR

    # 分层回测
    quantile_returns: pd.DataFrame  # 各分层累计收益 (index=日期, columns=分层号)
    long_short_return: float        # 多空组合年化收益（做多顶层,做空底层）
    monotonicity_score: float       # 单调性得分（-1 ~ 1）

    # 因子统计
    factor_coverage: float          # 因子覆盖率（有效值比例）
    factor_autocorr: float          # 因子自相关系数

    # 分层年化收益
    quantile_annualized: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> str:
        """格式化输出因子分析摘要"""
        lines = [
            "=" * 60,
            f"        因子分析报告: {self.factor_name}",
            "=" * 60,
            "",
            "【IC 分析】",
            f"  IC 均值:           {self.ic_mean:>10.4f}",
            f"  IC 标准差:         {self.ic_std:>10.4f}",
            f"  IC_IR:             {self.ic_ir:>10.4f}",
            f"  IC > 0 比例:       {self.ic_positive_ratio:>10.2%}",
            f"  Rank IC 均值:      {self.rank_ic_mean:>10.4f}",
            f"  Rank IC IR:        {self.rank_ic_ir:>10.4f}",
            "",
            "【分层回测】",
            f"  多空年化收益:      {self.long_short_return:>10.2%}",
            f"  单调性得分:        {self.monotonicity_score:>10.4f}",
            "",
            "【因子统计】",
            f"  因子覆盖率:        {self.factor_coverage:>10.2%}",
            f"  因子自相关系数:    {self.factor_autocorr:>10.4f}",
        ]

        if not self.quantile_annualized.empty:
            lines.append("")
            lines.append("【各分层年化收益】")
            for q, ret in self.quantile_annualized.items():
                lines.append(f"  第{q}层:            {ret:>10.2%}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class FactorAnalyzer:
    """因子分析器

    对因子进行 IC/IR 分析和分层回测，评估因子选股效果。

    Parameters
    ----------
    factor : Factor
        待分析的因子对象
    n_quantiles : int
        分层数量，默认 5（五分位）
    holding_period : int
        持有期（交易日），默认 20（月度调仓）

    Examples
    --------
    >>> from quant_trading.factor import MomentumFactor, FactorAnalyzer
    >>> factor = MomentumFactor(window=20)
    >>> analyzer = FactorAnalyzer(factor, n_quantiles=5)
    >>> result = analyzer.analyze(factor_data, return_data)
    """

    def __init__(
        self,
        factor: Factor,
        n_quantiles: int = 5,
        holding_period: int = 20,
    ) -> None:
        self.factor = factor
        self.n_quantiles = n_quantiles
        self.holding_period = holding_period

    def analyze(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
    ) -> FactorAnalysisResult:
        """执行完整的因子分析

        Parameters
        ----------
        factor_data : pd.DataFrame
            因子值面板，index 为日期，columns 为股票代码
        return_data : pd.DataFrame
            对应的收益率面板，index 为日期，columns 为股票代码。
            应为 **未来** holding_period 期间的收益率。

        Returns
        -------
        FactorAnalysisResult
            完整的因子分析结果
        """
        # 对齐数据
        common_dates = factor_data.index.intersection(return_data.index)
        common_stocks = factor_data.columns.intersection(return_data.columns)

        if len(common_dates) < 2 or len(common_stocks) < self.n_quantiles:
            raise ValueError(
                f"数据不足: {len(common_dates)} 期, {len(common_stocks)} 只股票，"
                f"需要至少 2 期和 {self.n_quantiles} 只股票"
            )

        factor_aligned = factor_data.loc[common_dates, common_stocks]
        return_aligned = return_data.loc[common_dates, common_stocks]

        # IC 分析
        ic_series = self._calc_ic_series(factor_aligned, return_aligned)
        rank_ic_series = self._calc_rank_ic_series(factor_aligned, return_aligned)

        # 分层回测
        quantile_returns = self._calc_quantile_returns(
            factor_aligned, return_aligned
        )

        # 多空组合收益
        long_short_return = self._calc_long_short_return(quantile_returns)

        # 单调性得分
        monotonicity = self._calc_monotonicity(quantile_returns)

        # 因子统计
        coverage = self._calc_coverage(factor_aligned)
        autocorr = self._calc_autocorrelation(factor_aligned)

        # 各层年化收益
        quantile_ann = self._calc_quantile_annualized(quantile_returns)

        # IC 统计
        ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
        ic_std = float(ic_series.std()) if len(ic_series) > 1 else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_positive_ratio = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0.0

        rank_ic_mean = float(rank_ic_series.mean()) if len(rank_ic_series) > 0 else 0.0
        rank_ic_std = float(rank_ic_series.std()) if len(rank_ic_series) > 1 else 0.0
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0

        return FactorAnalysisResult(
            factor_name=self.factor.name,
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_positive_ratio=ic_positive_ratio,
            rank_ic_series=rank_ic_series,
            rank_ic_mean=rank_ic_mean,
            rank_ic_ir=rank_ic_ir,
            quantile_returns=quantile_returns,
            long_short_return=long_short_return,
            monotonicity_score=monotonicity,
            factor_coverage=coverage,
            factor_autocorr=autocorr,
            quantile_annualized=quantile_ann,
        )

    # ------------------------------------------------------------------
    # IC 分析
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_ic_series(
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
    ) -> pd.Series:
        """计算每期截面 IC（Pearson 相关系数）

        IC_t = corr(factor_t, return_t)
        """
        ic_values = {}
        for date in factor_data.index:
            fv = factor_data.loc[date].dropna()
            rv = return_data.loc[date].dropna()
            common = fv.index.intersection(rv.index)
            if len(common) < 3:
                continue
            corr, _ = stats.pearsonr(fv[common], rv[common])
            if not np.isnan(corr):
                ic_values[date] = corr
        return pd.Series(ic_values, dtype=float)

    @staticmethod
    def _calc_rank_ic_series(
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
    ) -> pd.Series:
        """计算每期截面 Rank IC（Spearman 秩相关系数）

        Rank_IC_t = spearman_corr(factor_t, return_t)
        """
        ic_values = {}
        for date in factor_data.index:
            fv = factor_data.loc[date].dropna()
            rv = return_data.loc[date].dropna()
            common = fv.index.intersection(rv.index)
            if len(common) < 3:
                continue
            corr, _ = stats.spearmanr(fv[common], rv[common])
            if not np.isnan(corr):
                ic_values[date] = corr
        return pd.Series(ic_values, dtype=float)

    # ------------------------------------------------------------------
    # 分层回测
    # ------------------------------------------------------------------

    def _calc_quantile_returns(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算分层累计收益

        每期将股票按因子值排序分为 n_quantiles 组，
        计算每组的平均收益率，累计得到各层收益曲线。

        Returns
        -------
        pd.DataFrame
            index 为日期，columns 为分层号（1 ~ n_quantiles），
            值为累计收益率（从1开始的净值）
        """
        n_q = self.n_quantiles
        quantile_period_returns = {q: [] for q in range(1, n_q + 1)}
        dates = []

        for date in factor_data.index:
            fv = factor_data.loc[date].dropna()
            rv = return_data.loc[date].dropna()
            common = fv.index.intersection(rv.index)

            if len(common) < n_q:
                continue

            fv_sorted = fv[common].sort_values()
            # 分成 n_q 组
            group_size = len(fv_sorted) // n_q
            if group_size == 0:
                continue

            dates.append(date)
            for q in range(1, n_q + 1):
                if q < n_q:
                    start = (q - 1) * group_size
                    end = q * group_size
                else:
                    start = (q - 1) * group_size
                    end = len(fv_sorted)
                group_stocks = fv_sorted.index[start:end]
                avg_return = float(rv[group_stocks].mean())
                quantile_period_returns[q].append(avg_return)

        if not dates:
            return pd.DataFrame()

        # 构建 DataFrame
        period_df = pd.DataFrame(quantile_period_returns, index=dates)

        # 累计收益（净值曲线）
        cumulative = (1 + period_df).cumprod()
        return cumulative

    def _calc_long_short_return(
        self,
        quantile_returns: pd.DataFrame,
    ) -> float:
        """计算多空组合年化收益

        做多因子值最高组（第n_quantiles层），做空因子值最低组（第1层）
        """
        if quantile_returns.empty:
            return 0.0

        top_q = self.n_quantiles
        # 从净值反推总收益
        top_total = quantile_returns[top_q].iloc[-1] - 1
        bottom_total = quantile_returns[1].iloc[-1] - 1
        long_short_total = top_total - bottom_total

        n_periods = len(quantile_returns)
        if n_periods > 1:
            # 简化年化：假设每期为 holding_period 个交易日
            total_days = n_periods * self.holding_period
            annualized = (1 + long_short_total) ** (TRADING_DAYS_PER_YEAR / max(total_days, 1)) - 1
            return float(annualized)
        return float(long_short_total)

    def _calc_monotonicity(
        self,
        quantile_returns: pd.DataFrame,
    ) -> float:
        """计算各层收益的单调性得分

        使用 Spearman 秩相关衡量分层序号与层收益的单调关系。
        值接近 1 表示完美正向单调（因子值越高，收益越高）。
        值接近 -1 表示反向单调。
        """
        if quantile_returns.empty or len(quantile_returns.columns) < 3:
            return 0.0

        # 各层最终累计收益
        final_returns = quantile_returns.iloc[-1]
        quantile_indices = list(range(1, len(final_returns) + 1))

        corr, _ = stats.spearmanr(quantile_indices, final_returns.values)
        if np.isnan(corr):
            return 0.0
        return float(corr)

    def _calc_quantile_annualized(
        self,
        quantile_returns: pd.DataFrame,
    ) -> pd.Series:
        """计算各层年化收益"""
        if quantile_returns.empty:
            return pd.Series(dtype=float)

        n_periods = len(quantile_returns)
        total_days = n_periods * self.holding_period

        result = {}
        for q in quantile_returns.columns:
            total_ret = quantile_returns[q].iloc[-1] - 1
            if total_days > 0:
                ann = (1 + total_ret) ** (TRADING_DAYS_PER_YEAR / total_days) - 1
            else:
                ann = 0.0
            result[q] = float(ann)

        return pd.Series(result, dtype=float)

    # ------------------------------------------------------------------
    # 因子统计
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_coverage(factor_data: pd.DataFrame) -> float:
        """计算因子覆盖率（有效值占比）"""
        total = factor_data.size
        if total == 0:
            return 0.0
        valid = factor_data.notna().sum().sum()
        return float(valid / total)

    @staticmethod
    def _calc_autocorrelation(factor_data: pd.DataFrame) -> float:
        """计算因子截面排名的自相关系数

        衡量因子的稳定性：每期因子排名与上期排名的相关性。
        """
        if len(factor_data) < 2:
            return 0.0

        rank_corrs = []
        dates = factor_data.index
        for i in range(1, len(dates)):
            prev = factor_data.iloc[i - 1].dropna().rank()
            curr = factor_data.iloc[i].dropna().rank()
            common = prev.index.intersection(curr.index)
            if len(common) < 3:
                continue
            corr, _ = stats.spearmanr(prev[common], curr[common])
            if not np.isnan(corr):
                rank_corrs.append(corr)

        if not rank_corrs:
            return 0.0
        return float(np.mean(rank_corrs))
