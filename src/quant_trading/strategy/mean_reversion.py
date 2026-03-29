"""均值回归策略 (Mean Reversion)

价格偏离移动平均线过远时回归交易。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.trend import SMA
from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("mean_reversion")
class MeanReversionStrategy(Strategy):
    """均值回归策略。

    原理: 价格偏离移动平均线过远时回归
    - 计算 Z-Score = (price - SMA) / rolling_std
    - Z-Score < -entry_threshold: 买入 (超卖回归)
    - Z-Score > entry_threshold: 卖出 (超买回归)
    - |Z-Score| < exit_threshold: 平仓 (hold)

    Parameters
    ----------
    window : int
        移动平均和标准差的滚动窗口，默认 20。
    entry_threshold : float
        开仓阈值（Z-Score 绝对值），默认 2.0。
    exit_threshold : float
        平仓阈值（Z-Score 绝对值），默认 0.5。
    """

    def __init__(
        self,
        window: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if window < 2:
            raise ValueError(f"window 必须 >= 2，收到 {window}")
        if entry_threshold <= 0:
            raise ValueError(f"entry_threshold 必须 > 0，收到 {entry_threshold}")
        if exit_threshold < 0:
            raise ValueError(f"exit_threshold 必须 >= 0，收到 {exit_threshold}")
        if exit_threshold >= entry_threshold:
            raise ValueError(
                f"exit_threshold ({exit_threshold}) 必须小于 "
                f"entry_threshold ({entry_threshold})"
            )
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self._sma = SMA(window=window)

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 计算 SMA（如果尚未存在）
        sma_col = f"sma_{self.window}"
        if sma_col not in data.columns:
            data = self._sma.calculate(data)

        # 滚动标准差
        rolling_std = data["close"].rolling(
            window=self.window, min_periods=self.window,
        ).std(ddof=0)

        # Z-Score
        data["mr_zscore"] = (data["close"] - data[sma_col]) / rolling_std

        # 生成信号
        data["signal"] = "hold"
        data.loc[data["mr_zscore"] < -self.entry_threshold, "signal"] = "buy"
        data.loc[data["mr_zscore"] > self.entry_threshold, "signal"] = "sell"
        # |Z-Score| < exit_threshold 时平仓（保持 hold）

        return data
