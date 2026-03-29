"""复合策略 (Composite Strategy)

将多个子策略的信号进行投票合并，支持多数、全体一致、任意三种投票模式。
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

import pandas as pd

from quant_trading.strategy.base import Strategy, StrategyRegistry


VotingMethod = Literal["majority", "unanimous", "any"]

_VALID_VOTING_METHODS: set[str] = {"majority", "unanimous", "any"}


@StrategyRegistry.register("composite")
class CompositeStrategy(Strategy):
    """复合策略 — 组合多个子策略，通过投票决定最终信号。

    Parameters
    ----------
    strategies : list[Strategy]
        参与投票的子策略列表（至少 1 个）。
    voting : {"majority", "unanimous", "any"}
        投票方法:
        - ``"majority"``: 超过半数策略同意时发出信号。
        - ``"unanimous"``: 全部策略一致时发出信号。
        - ``"any"``: 任一策略发出信号即生效。
    name : str
        策略名称。
    """

    def __init__(
        self,
        strategies: list[Strategy] | None = None,
        voting: VotingMethod = "majority",
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if not strategies:
            raise ValueError("复合策略至少需要 1 个子策略")
        if voting not in _VALID_VOTING_METHODS:
            raise ValueError(
                f"voting 必须为 {_VALID_VOTING_METHODS} 之一，收到 {voting!r}"
            )
        self.strategies = strategies
        self.voting: VotingMethod = voting

    def get_params(self) -> dict:
        return {
            "voting": self.voting,
            "sub_strategies": [
                {"name": s.name, "params": s.get_params()} for s in self.strategies
            ],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 收集各子策略的信号
        signal_columns: list[str] = []
        for i, strategy in enumerate(self.strategies):
            col = f"_signal_{i}_{strategy.name}"
            # 在副本上计算以避免子策略间互相污染指标列
            sub_data = strategy.generate_signals(data.copy())
            data[col] = sub_data["signal"]
            signal_columns.append(col)

        n_strategies = len(self.strategies)

        # 逐行投票
        def _vote(row: pd.Series) -> str:
            signals = [row[c] for c in signal_columns]
            counts = Counter(signals)

            buy_count = counts.get("buy", 0)
            sell_count = counts.get("sell", 0)

            if self.voting == "majority":
                threshold = n_strategies / 2.0
                if buy_count > threshold:
                    return "buy"
                if sell_count > threshold:
                    return "sell"
                return "hold"

            elif self.voting == "unanimous":
                if buy_count == n_strategies:
                    return "buy"
                if sell_count == n_strategies:
                    return "sell"
                return "hold"

            else:  # "any"
                # 买卖同时出现时，优先级: buy > sell > hold
                if buy_count > 0:
                    return "buy"
                if sell_count > 0:
                    return "sell"
                return "hold"

        data["signal"] = data.apply(_vote, axis=1)

        # 清理临时列
        data.drop(columns=signal_columns, inplace=True)
        return data
