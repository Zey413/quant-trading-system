"""基准管理器 - 获取指数数据并计算相对基准的绩效指标

提供：
- 沪深300、上证50、中证500等常见指数数据获取
- Alpha / Beta 计算
- 信息比率 (IR)
- 超额收益序列
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_trading.backtest.metrics import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

# 常见A股指数代码映射 (AKShare 指数代码)
INDEX_CODE_MAP: dict[str, str] = {
    "沪深300": "sh000300",
    "hs300": "sh000300",
    "上证50": "sh000016",
    "sz50": "sh000016",
    "中证500": "sh000905",
    "zz500": "sh000905",
    "上证指数": "sh000001",
    "szzs": "sh000001",
    "深证成指": "sz399001",
    "szcz": "sz399001",
    "创业板指": "sz399006",
    "cybz": "sz399006",
}


@dataclass
class BenchmarkResult:
    """基准比较结果"""

    benchmark_name: str           # 基准名称
    benchmark_return: float       # 基准总收益率
    benchmark_annualized: float   # 基准年化收益率

    alpha: float                  # Jensen's Alpha（年化）
    beta: float                   # Beta系数
    information_ratio: float      # 信息比率 (IR)
    tracking_error: float         # 跟踪误差（年化）

    excess_return: float          # 累计超额收益
    excess_annualized: float      # 年化超额收益

    excess_curve: pd.Series       # 累计超额收益曲线
    benchmark_curve: pd.Series    # 基准净值曲线

    def summary(self) -> str:
        """格式化输出基准比较摘要"""
        lines = [
            "=" * 60,
            f"          基准比较报告 ({self.benchmark_name})",
            "=" * 60,
            "",
            "【基准表现】",
            f"  基准总收益率:      {self.benchmark_return:>10.2%}",
            f"  基准年化收益率:    {self.benchmark_annualized:>10.2%}",
            "",
            "【相对指标】",
            f"  Alpha (年化):      {self.alpha:>10.4f}",
            f"  Beta:              {self.beta:>10.4f}",
            f"  信息比率 (IR):     {self.information_ratio:>10.4f}",
            f"  跟踪误差 (年化):  {self.tracking_error:>10.2%}",
            "",
            "【超额收益】",
            f"  累计超额收益:      {self.excess_return:>10.2%}",
            f"  年化超额收益:      {self.excess_annualized:>10.2%}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class BenchmarkManager:
    """基准管理器

    获取指数数据，计算策略相对于基准的 Alpha、Beta、IR 和超额收益。

    Parameters
    ----------
    benchmark_name : str
        基准名称或代码，支持中文名 (如 "沪深300") 或英文简写 (如 "hs300")
    benchmark_data : pd.DataFrame | pd.Series | None
        预加载的基准数据。若为 None，则通过 ``fetch_benchmark`` 获取。
        如果传入 DataFrame，应包含 ``close`` 列；
        如果传入 Series，应为收盘价序列，index 为日期。

    Examples
    --------
    >>> bm = BenchmarkManager("沪深300")
    >>> # 使用预加载的数据
    >>> bm = BenchmarkManager("hs300", benchmark_data=index_close_series)
    >>> result = bm.compare(strategy_equity_curve)
    """

    def __init__(
        self,
        benchmark_name: str = "沪深300",
        benchmark_data: pd.DataFrame | pd.Series | None = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self._benchmark_close: pd.Series | None = None

        if benchmark_data is not None:
            self._benchmark_close = self._normalize_data(benchmark_data)

    @staticmethod
    def _normalize_data(data: pd.DataFrame | pd.Series) -> pd.Series:
        """将输入数据标准化为收盘价 Series"""
        if isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                series = data["close"]
            elif "Close" in data.columns:
                series = data["Close"]
            else:
                raise ValueError("DataFrame 必须包含 'close' 或 'Close' 列")
        elif isinstance(data, pd.Series):
            series = data
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

        series = series.copy()
        series.index = pd.to_datetime(series.index)
        return series.sort_index()

    @staticmethod
    def resolve_index_code(name: str) -> str:
        """将中文名/英文简写解析为 AKShare 指数代码

        Parameters
        ----------
        name : str
            基准名称，支持中文名或英文简写

        Returns
        -------
        str
            AKShare 格式的指数代码
        """
        return INDEX_CODE_MAP.get(name, name)

    def fetch_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """通过 AKShare 获取指数日线数据

        Parameters
        ----------
        start_date : str
            开始日期 (YYYYMMDD 或 YYYY-MM-DD)
        end_date : str
            结束日期

        Returns
        -------
        pd.Series
            收盘价序列，index 为日期
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError(
                "获取指数数据需要 akshare，请运行: pip install akshare"
            )

        index_code = self.resolve_index_code(self.benchmark_name)
        # 去掉前缀 sh/sz
        pure_code = index_code.replace("sh", "").replace("sz", "")

        # 标准化日期格式
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")

        logger.info(
            "获取指数数据: %s (%s), %s ~ %s",
            self.benchmark_name, pure_code, start_clean, end_clean,
        )

        df = ak.stock_zh_index_daily(symbol=index_code)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.loc[start_clean:end_clean]

        if df.empty:
            raise ValueError(
                f"未获取到 {self.benchmark_name} ({index_code}) 在 "
                f"{start_clean}~{end_clean} 的数据"
            )

        self._benchmark_close = df["close"]
        return self._benchmark_close.copy()

    @property
    def benchmark_close(self) -> pd.Series:
        """基准收盘价序列"""
        if self._benchmark_close is None:
            raise ValueError(
                "基准数据未加载，请先调用 fetch_benchmark() 或在初始化时传入数据"
            )
        return self._benchmark_close

    def compare(
        self,
        equity_curve: pd.Series,
        risk_free_rate: float = 0.03,
    ) -> BenchmarkResult:
        """将策略净值曲线与基准进行比较

        Parameters
        ----------
        equity_curve : pd.Series
            策略净值曲线，index 为日期
        risk_free_rate : float
            年化无风险利率，默认 3%

        Returns
        -------
        BenchmarkResult
            包含 Alpha、Beta、IR 等指标的结果
        """
        benchmark = self.benchmark_close

        # 对齐日期
        equity_curve = equity_curve.copy()
        equity_curve.index = pd.to_datetime(equity_curve.index)

        common_dates = equity_curve.index.intersection(benchmark.index)
        if len(common_dates) < 2:
            # 如果无法按日期对齐（例如测试数据），尝试按长度截取
            min_len = min(len(equity_curve), len(benchmark))
            strat_vals = equity_curve.iloc[:min_len]
            bench_vals = benchmark.iloc[:min_len]
        else:
            common_dates = common_dates.sort_values()
            strat_vals = equity_curve.loc[common_dates]
            bench_vals = benchmark.loc[common_dates]

        # 归一化为净值（从1开始）
        strat_nav = strat_vals / strat_vals.iloc[0]
        bench_nav = bench_vals / bench_vals.iloc[0]

        # 日收益率
        strat_returns = strat_nav.pct_change().dropna()
        bench_returns = bench_nav.pct_change().dropna()

        # 对齐收益率序列
        aligned_idx = strat_returns.index.intersection(bench_returns.index)
        if len(aligned_idx) < 2:
            min_len = min(len(strat_returns), len(bench_returns))
            strat_returns = strat_returns.iloc[:min_len]
            bench_returns = bench_returns.iloc[:min_len]
        else:
            strat_returns = strat_returns.loc[aligned_idx]
            bench_returns = bench_returns.loc[aligned_idx]

        # ---- 基准收益 ----
        benchmark_return = float(bench_nav.iloc[-1] / bench_nav.iloc[0] - 1)
        trading_days = len(bench_nav)
        if trading_days > 1:
            benchmark_annualized = (
                (1 + benchmark_return) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1
            )
        else:
            benchmark_annualized = 0.0

        # ---- Beta ----
        beta = self._calc_beta(strat_returns, bench_returns)

        # ---- Alpha (Jensen's Alpha, 年化) ----
        strat_total = float(strat_nav.iloc[-1] / strat_nav.iloc[0] - 1)
        if trading_days > 1:
            strat_annualized = (
                (1 + strat_total) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1
            )
        else:
            strat_annualized = 0.0
        alpha = self._calc_alpha(
            strat_annualized, benchmark_annualized, beta, risk_free_rate
        )

        # ---- 超额收益 ----
        excess_daily = strat_returns.values - bench_returns.values
        excess_curve = pd.Series(
            (1 + pd.Series(excess_daily)).cumprod() - 1,
            index=strat_returns.index,
        )
        excess_return = float(strat_total - benchmark_return)
        if trading_days > 1:
            excess_annualized = strat_annualized - benchmark_annualized
        else:
            excess_annualized = 0.0

        # ---- 跟踪误差 & 信息比率 ----
        tracking_error = self._calc_tracking_error(excess_daily)
        information_ratio = self._calc_information_ratio(
            excess_annualized, tracking_error
        )

        return BenchmarkResult(
            benchmark_name=self.benchmark_name,
            benchmark_return=benchmark_return,
            benchmark_annualized=benchmark_annualized,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            excess_return=excess_return,
            excess_annualized=excess_annualized,
            excess_curve=excess_curve,
            benchmark_curve=bench_nav,
        )

    # ------------------------------------------------------------------
    # 内部计算方法
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_beta(
        strat_returns: pd.Series,
        bench_returns: pd.Series,
    ) -> float:
        """计算 Beta 系数

        Beta = Cov(Rp, Rm) / Var(Rm)
        """
        if len(strat_returns) < 2:
            return 0.0
        cov_matrix = np.cov(
            strat_returns.values.astype(float),
            bench_returns.values.astype(float),
        )
        var_bench = cov_matrix[1, 1]
        if var_bench == 0:
            return 0.0
        return float(cov_matrix[0, 1] / var_bench)

    @staticmethod
    def _calc_alpha(
        strat_annualized: float,
        bench_annualized: float,
        beta: float,
        risk_free_rate: float,
    ) -> float:
        """计算 Jensen's Alpha (年化)

        Alpha = Rp - [Rf + Beta * (Rm - Rf)]
        """
        expected = risk_free_rate + beta * (bench_annualized - risk_free_rate)
        return strat_annualized - expected

    @staticmethod
    def _calc_tracking_error(excess_daily: np.ndarray) -> float:
        """计算年化跟踪误差

        TE = std(超额日收益) * sqrt(244)
        """
        if len(excess_daily) < 2:
            return 0.0
        return float(np.std(excess_daily, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _calc_information_ratio(
        excess_annualized: float,
        tracking_error: float,
    ) -> float:
        """计算信息比率 (IR)

        IR = 年化超额收益 / 年化跟踪误差
        """
        if tracking_error == 0:
            return 0.0
        return excess_annualized / tracking_error
