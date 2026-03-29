"""止损/止盈管理模块

提供基于持仓成本的止损止盈检查：
- 固定百分比止损：当前价 <= 成本 * (1 - stop_loss_pct)
- 固定百分比止盈：当前价 >= 成本 * (1 + take_profit_pct)
"""

from __future__ import annotations

import logging

from quant_trading.core.config import RiskConfig
from quant_trading.core.models import Position

logger = logging.getLogger(__name__)


class StopLossManager:
    """止损/止盈管理器

    基于持仓均价和配置的百分比阈值，判断是否触发止损或止盈。

    Attributes:
        config: 风险配置
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def check_stop_loss(self, position: Position) -> bool:
        """检查是否触发止损

        止损条件：当前价 <= 平均成本 * (1 - stop_loss_pct)

        Args:
            position: 持仓对象

        Returns:
            True表示触发止损
        """
        if position.avg_cost <= 0 or position.quantity <= 0:
            return False

        stop_price = position.avg_cost * (1 - self.config.stop_loss_pct)
        triggered = position.current_price <= stop_price

        if triggered:
            logger.info(
                "触发止损: %s, 成本 %.4f, 当前价 %.4f, 止损价 %.4f, 亏损 %.2f%%",
                position.symbol,
                position.avg_cost,
                position.current_price,
                stop_price,
                position.unrealized_pnl_pct * 100,
            )

        return triggered

    def check_take_profit(self, position: Position) -> bool:
        """检查是否触发止盈

        止盈条件：当前价 >= 平均成本 * (1 + take_profit_pct)

        Args:
            position: 持仓对象

        Returns:
            True表示触发止盈
        """
        if position.avg_cost <= 0 or position.quantity <= 0:
            return False

        profit_price = position.avg_cost * (1 + self.config.take_profit_pct)
        triggered = position.current_price >= profit_price

        if triggered:
            logger.info(
                "触发止盈: %s, 成本 %.4f, 当前价 %.4f, 止盈价 %.4f, 盈利 %.2f%%",
                position.symbol,
                position.avg_cost,
                position.current_price,
                profit_price,
                position.unrealized_pnl_pct * 100,
            )

        return triggered

    def should_exit(self, position: Position) -> tuple[bool, str]:
        """检查是否应该退出持仓

        依次检查止损和止盈条件。

        Args:
            position: 持仓对象

        Returns:
            (should_exit, reason): 是否退出及原因
        """
        if self.check_stop_loss(position):
            return (
                True,
                f"止损: {position.symbol} 当前价 {position.current_price:.4f}"
                f" <= 止损价 {position.avg_cost * (1 - self.config.stop_loss_pct):.4f}"
                f" (亏损 {position.unrealized_pnl_pct:.2%})",
            )

        if self.check_take_profit(position):
            return (
                True,
                f"止盈: {position.symbol} 当前价 {position.current_price:.4f}"
                f" >= 止盈价 {position.avg_cost * (1 + self.config.take_profit_pct):.4f}"
                f" (盈利 {position.unrealized_pnl_pct:.2%})",
            )

        return False, ""

    def get_stop_price(self, position: Position) -> float:
        """获取止损价格

        Args:
            position: 持仓对象

        Returns:
            止损触发价格
        """
        return position.avg_cost * (1 - self.config.stop_loss_pct)

    def get_take_profit_price(self, position: Position) -> float:
        """获取止盈价格

        Args:
            position: 持仓对象

        Returns:
            止盈触发价格
        """
        return position.avg_cost * (1 + self.config.take_profit_pct)


# ---------------------------------------------------------------------------
# Trailing Stop（追踪止损）
# ---------------------------------------------------------------------------

class TrailingStopMode:
    """追踪止损模式常量"""
    PERCENT = "percent"       # 固定百分比追踪
    ATR = "atr"               # ATR倍数追踪
    STEP = "step"             # 阶梯式追踪


class TrailingStopManager:
    """追踪止损管理器

    跟踪持仓期间的价格最高点，动态上移止损价。支持三种模式：
    - 固定百分比追踪：止损价 = 最高价 × (1 - trail_pct)
    - ATR倍数追踪：止损价 = 最高价 - atr_multiplier × ATR
    - 阶梯式追踪：盈利越多，止损比例越紧

    Attributes:
        mode: 追踪模式 ("percent" / "atr" / "step")
        trail_pct: 固定百分比模式下的回撤百分比 (默认 0.05 = 5%)
        atr_multiplier: ATR模式下的倍数 (默认 2.0)
        steps: 阶梯模式下的阶梯定义 [(盈利阈值, 止损百分比), ...]
    """

    # 默认阶梯：盈利越多，止损越紧
    DEFAULT_STEPS: list[tuple[float, float]] = [
        (0.00, 0.05),   # 盈利 0%~5%：止损回撤 5%
        (0.05, 0.04),   # 盈利 5%~10%：止损回撤 4%
        (0.10, 0.03),   # 盈利 10%~20%：止损回撤 3%
        (0.20, 0.02),   # 盈利 20%+：止损回撤 2%
    ]

    def __init__(
        self,
        mode: str = TrailingStopMode.PERCENT,
        trail_pct: float = 0.05,
        atr_multiplier: float = 2.0,
        steps: list[tuple[float, float]] | None = None,
    ) -> None:
        self.mode = mode
        self.trail_pct = trail_pct
        self.atr_multiplier = atr_multiplier
        self.steps = steps or self.DEFAULT_STEPS
        # symbol -> 历史最高价
        self._high_watermarks: dict[str, float] = {}
        # symbol -> 当前追踪止损价
        self._stop_prices: dict[str, float] = {}

    def reset(self, symbol: str | None = None) -> None:
        """重置追踪状态

        Args:
            symbol: 指定股票代码，None则重置全部
        """
        if symbol is None:
            self._high_watermarks.clear()
            self._stop_prices.clear()
        else:
            self._high_watermarks.pop(symbol, None)
            self._stop_prices.pop(symbol, None)

    def update(
        self,
        symbol: str,
        current_price: float,
        avg_cost: float,
        atr: float | None = None,
    ) -> float:
        """更新追踪止损价

        每个交易日调用一次，更新最高价和止损价。止损价只会上移，不会下移。

        Args:
            symbol: 股票代码
            current_price: 当前价格
            avg_cost: 持仓均价（阶梯模式需要）
            atr: ATR值（ATR模式需要）

        Returns:
            当前追踪止损价
        """
        # 更新最高价
        prev_high = self._high_watermarks.get(symbol, current_price)
        new_high = max(prev_high, current_price)
        self._high_watermarks[symbol] = new_high

        # 计算新止损价
        if self.mode == TrailingStopMode.PERCENT:
            new_stop = new_high * (1 - self.trail_pct)

        elif self.mode == TrailingStopMode.ATR:
            if atr is None or atr <= 0:
                logger.warning(
                    "ATR追踪止损需要有效的ATR值，%s 使用百分比回退", symbol
                )
                new_stop = new_high * (1 - self.trail_pct)
            else:
                new_stop = new_high - self.atr_multiplier * atr

        elif self.mode == TrailingStopMode.STEP:
            # 计算当前盈利百分比
            profit_pct = (new_high - avg_cost) / avg_cost if avg_cost > 0 else 0
            # 找到对应的阶梯
            trail = self.trail_pct  # 默认
            for threshold, pct in sorted(self.steps, key=lambda x: x[0], reverse=True):
                if profit_pct >= threshold:
                    trail = pct
                    break
            new_stop = new_high * (1 - trail)

        else:
            raise ValueError(f"未知的追踪止损模式: {self.mode}")

        # 止损价只上移，不下移
        prev_stop = self._stop_prices.get(symbol, 0.0)
        final_stop = max(prev_stop, new_stop)
        self._stop_prices[symbol] = final_stop

        return final_stop

    def check(
        self,
        symbol: str,
        current_price: float,
        avg_cost: float,
        atr: float | None = None,
    ) -> tuple[bool, float]:
        """检查是否触发追踪止损

        先更新止损价，再判断当前价是否低于止损价。

        Args:
            symbol: 股票代码
            current_price: 当前价格
            avg_cost: 持仓均价
            atr: ATR值（ATR模式需要）

        Returns:
            (triggered, stop_price): 是否触发及当前止损价
        """
        stop_price = self.update(symbol, current_price, avg_cost, atr)
        triggered = current_price <= stop_price

        if triggered:
            high = self._high_watermarks.get(symbol, current_price)
            drawdown = (high - current_price) / high if high > 0 else 0
            logger.info(
                "触发追踪止损: %s, 最高价 %.4f, 止损价 %.4f, "
                "当前价 %.4f, 回撤 %.2f%%",
                symbol, high, stop_price, current_price, drawdown * 100,
            )

        return triggered, stop_price

    def get_stop_price(self, symbol: str) -> float | None:
        """获取当前追踪止损价

        Args:
            symbol: 股票代码

        Returns:
            止损价，如果无记录返回None
        """
        return self._stop_prices.get(symbol)

    def get_high_watermark(self, symbol: str) -> float | None:
        """获取历史最高价

        Args:
            symbol: 股票代码

        Returns:
            最高价，如果无记录返回None
        """
        return self._high_watermarks.get(symbol)
