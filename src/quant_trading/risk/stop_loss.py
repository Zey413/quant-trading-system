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
