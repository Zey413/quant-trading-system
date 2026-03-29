"""风险管理器 - 统一风控入口

整合仓位管理和止损止盈管理，提供：
- 信号验证：检查是否超过持仓限制
- 仓位计算：根据配置方法计算下单数量
- 退出检查：遍历持仓检查止损/止盈
"""

from __future__ import annotations

import logging
from datetime import date

from quant_trading.core.config import RiskConfig
from quant_trading.core.enums import SignalType
from quant_trading.core.models import Portfolio, Position, Signal
from quant_trading.risk.position_sizer import PositionSizer
from quant_trading.risk.stop_loss import StopLossManager

logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器

    统一的风控管理入口，协调仓位管理和止损止盈。

    Attributes:
        config: 风险配置
        position_sizer: 仓位管理器
        stop_loss_manager: 止损止盈管理器
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.position_sizer = PositionSizer(config)
        self.stop_loss_manager = StopLossManager(config)

    def validate_signal(
        self,
        signal: Signal,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """验证交易信号是否满足风控要求

        买入信号检查：
        1. 单只股票持仓不超过max_position_pct
        2. 总持仓不超过max_total_position_pct

        卖出信号：默认通过（卖出减仓总是被允许的）。

        Args:
            signal: 交易信号
            portfolio: 当前组合

        Returns:
            (is_valid, reason): 是否通过验证及原因
        """
        if signal.is_sell():
            return True, "卖出信号通过验证"

        if not signal.is_buy():
            return True, "非交易信号通过验证"

        total_value = portfolio.total_value
        if total_value <= 0:
            return False, "组合总值为零"

        # 检查单只股票持仓比例
        position = portfolio.get_position(signal.symbol)
        if position is not None:
            position_pct = position.market_value / total_value
            if position_pct >= self.config.max_position_pct:
                return (
                    False,
                    f"{signal.symbol} 持仓比例 {position_pct:.2%} "
                    f"已达上限 {self.config.max_position_pct:.2%}",
                )

        # 检查总持仓比例
        total_position_pct = portfolio.market_value / total_value
        if total_position_pct >= self.config.max_total_position_pct:
            return (
                False,
                f"总持仓比例 {total_position_pct:.2%} "
                f"已达上限 {self.config.max_total_position_pct:.2%}",
            )

        return True, "验证通过"

    def calculate_order_quantity(
        self,
        signal: Signal,
        portfolio: Portfolio,
    ) -> int:
        """计算下单数量

        委托给PositionSizer计算。

        Args:
            signal: 交易信号
            portfolio: 当前组合

        Returns:
            下单股数（已取整到lot_size）
        """
        return self.position_sizer.calculate_quantity(signal, portfolio)

    def check_exits(self, portfolio: Portfolio) -> list[Signal]:
        """检查所有持仓的止损/止盈

        遍历组合中的每个持仓，检查是否触发止损或止盈条件。
        如果触发，生成卖出信号。

        Args:
            portfolio: 当前组合

        Returns:
            需要退出的卖出信号列表
        """
        exit_signals: list[Signal] = []

        for symbol, position in list(portfolio.positions.items()):
            if position.quantity <= 0:
                continue
            if position.available_quantity <= 0:
                # T+1规则：当日买入不可卖出
                continue

            should_exit, reason = self.stop_loss_manager.should_exit(position)

            if should_exit:
                exit_signal = Signal(
                    date=date.today(),  # 实际日期由引擎覆盖
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=position.current_price,
                    strength=1.0,
                    strategy_name="risk_manager",
                    reason=reason,
                )
                exit_signals.append(exit_signal)
                logger.info("触发退出信号: %s - %s", symbol, reason)

        return exit_signals

    def update_kelly_params(
        self,
        win_rate: float,
        profit_loss_ratio: float,
    ) -> None:
        """更新Kelly公式参数

        Args:
            win_rate: 历史胜率
            profit_loss_ratio: 盈亏比
        """
        self.position_sizer.update_kelly_params(win_rate, profit_loss_ratio)

    def update_atr(self, atr: float) -> None:
        """更新ATR值

        Args:
            atr: 最新ATR
        """
        self.position_sizer.update_atr(atr)

    def get_risk_report(self, portfolio: Portfolio) -> dict:
        """获取风险报告

        Args:
            portfolio: 当前组合

        Returns:
            风险状态报告字典
        """
        total_value = portfolio.total_value
        if total_value <= 0:
            return {"error": "组合总值为零"}

        positions_risk = {}
        for symbol, position in portfolio.positions.items():
            if position.quantity <= 0:
                continue

            position_pct = position.market_value / total_value
            should_exit, reason = self.stop_loss_manager.should_exit(position)

            positions_risk[symbol] = {
                "position_pct": round(position_pct, 4),
                "unrealized_pnl_pct": round(position.unrealized_pnl_pct, 4),
                "stop_loss_price": round(
                    self.stop_loss_manager.get_stop_price(position), 4
                ),
                "take_profit_price": round(
                    self.stop_loss_manager.get_take_profit_price(position), 4
                ),
                "should_exit": should_exit,
                "exit_reason": reason,
            }

        total_position_pct = portfolio.market_value / total_value

        return {
            "total_value": round(total_value, 2),
            "cash": round(portfolio.cash, 2),
            "total_position_pct": round(total_position_pct, 4),
            "max_total_position_pct": self.config.max_total_position_pct,
            "positions_risk": positions_risk,
        }
