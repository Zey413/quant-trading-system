"""投资组合管理模块

负责管理回测过程中的组合状态：
- 现金管理
- 持仓更新（买入增仓/卖出减仓）
- T+1规则下的可卖数量管理
- 盈亏计算
- 组合快照
"""

from __future__ import annotations

import logging
import uuid
from datetime import date

from quant_trading.core.enums import OrderSide, OrderStatus
from quant_trading.core.models import Order, Portfolio, Position, TradeRecord

logger = logging.getLogger(__name__)


class PortfolioManager:
    """投资组合管理器

    管理回测过程中组合的所有状态变更，包括买入建仓、卖出平仓、
    价格更新和T+1规则处理。

    Attributes:
        portfolio: 当前组合状态
        _pending_available: 记录当日买入的持仓，下一交易日才可卖出
    """

    def __init__(self, initial_capital: float) -> None:
        """初始化组合管理器

        Args:
            initial_capital: 初始资金
        """
        self.portfolio = Portfolio(
            cash=initial_capital,
            positions={},
            initial_capital=initial_capital,
        )
        # 记录当日买入的数量，key为symbol，value为当日买入的股数
        # 在new_trading_day()时将这些数量转入available_quantity
        self._pending_available: dict[str, int] = {}

    def process_buy(self, order: Order) -> None:
        """处理买入成交

        更新现金、新建或加仓持仓。当日买入的部分不计入可卖数量（T+1）。

        Args:
            order: 已成交的买入订单
        """
        if order.status != OrderStatus.FILLED:
            logger.warning("订单 %s 未成交，无法处理买入", order.order_id)
            return
        if order.side != OrderSide.BUY:
            logger.warning("订单 %s 不是买入订单", order.order_id)
            return

        # 计算总花费 = 成交额 + 佣金（买入无印花税）
        total_cost = (
            order.filled_price * order.filled_quantity
            + order.commission
            + order.tax
        )

        # 扣减现金
        self.portfolio.cash -= total_cost

        symbol = order.symbol
        position = self.portfolio.get_position(symbol)

        if position is not None and position.quantity > 0:
            # 加仓：更新平均成本
            old_total = position.avg_cost * position.quantity
            new_total = order.filled_price * order.filled_quantity
            total_quantity = position.quantity + order.filled_quantity
            position.avg_cost = (old_total + new_total) / total_quantity
            position.quantity = total_quantity
            # 加仓的部分不立即可卖（T+1）
        else:
            # 新建仓位
            position = Position(
                symbol=symbol,
                quantity=order.filled_quantity,
                available_quantity=0,  # T+1：当日买入不可卖
                avg_cost=order.filled_price,
                current_price=order.filled_price,
                buy_date=order.date,
            )
            self.portfolio.positions[symbol] = position

        # 记录待转入可卖的数量
        self._pending_available[symbol] = (
            self._pending_available.get(symbol, 0) + order.filled_quantity
        )

        logger.debug(
            "买入 %s %d股 @ %.4f, 花费 %.2f, 剩余现金 %.2f",
            symbol,
            order.filled_quantity,
            order.filled_price,
            total_cost,
            self.portfolio.cash,
        )

    def process_sell(self, order: Order) -> TradeRecord:
        """处理卖出成交

        更新现金、减仓或清仓，计算平仓盈亏。

        Args:
            order: 已成交的卖出订单

        Returns:
            TradeRecord: 交易记录（含平仓盈亏）

        Raises:
            ValueError: 订单未成交或持仓不足
        """
        if order.status != OrderStatus.FILLED:
            raise ValueError(f"订单 {order.order_id} 未成交，无法处理卖出")
        if order.side != OrderSide.SELL:
            raise ValueError(f"订单 {order.order_id} 不是卖出订单")

        symbol = order.symbol
        position = self.portfolio.get_position(symbol)
        if position is None or position.quantity <= 0:
            raise ValueError(f"未持有 {symbol}，无法卖出")

        # 卖出收入 = 成交额 - 佣金 - 印花税
        turnover = order.filled_price * order.filled_quantity
        net_income = turnover - order.commission - order.tax

        # 加回现金
        self.portfolio.cash += net_income

        # 计算平仓盈亏（卖出价 - 买入均价）* 数量 - 交易费用
        pnl = (
            (order.filled_price - position.avg_cost) * order.filled_quantity
            - order.commission
            - order.tax
        )

        # 更新持仓
        position.quantity -= order.filled_quantity
        position.available_quantity -= order.filled_quantity

        # 同步更新pending_available中的数量（如果当日也买入了该股票）
        if symbol in self._pending_available:
            # 优先从available_quantity扣减（即之前日子买的），
            # 但available_quantity可能被扣到负数说明pending也要扣减
            if position.available_quantity < 0:
                # 这种情况不应该发生（broker已校验），做防守处理
                self._pending_available[symbol] = max(
                    0,
                    self._pending_available[symbol] + position.available_quantity,
                )
                position.available_quantity = 0

        # 如果持仓清零，移除
        if position.quantity <= 0:
            del self.portfolio.positions[symbol]
            self._pending_available.pop(symbol, None)

        # 生成交易记录
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            date=order.date,
            symbol=symbol,
            side="sell",
            quantity=order.filled_quantity,
            price=order.filled_price,
            commission=order.commission,
            tax=order.tax,
            pnl=round(pnl, 2),
        )

        logger.debug(
            "卖出 %s %d股 @ %.4f, 收入 %.2f, PnL=%.2f, 剩余现金 %.2f",
            symbol,
            order.filled_quantity,
            order.filled_price,
            net_income,
            pnl,
            self.portfolio.cash,
        )

        return trade

    def update_prices(self, prices: dict[str, float]) -> None:
        """更新持仓的当前价格

        Args:
            prices: {symbol: current_price} 字典
        """
        for symbol, price in prices.items():
            position = self.portfolio.get_position(symbol)
            if position is not None:
                position.current_price = price

    def new_trading_day(self) -> None:
        """处理新交易日

        T+1规则：将前一日买入的数量转入可卖数量。
        每个新交易日开盘前调用。
        """
        for symbol, qty in self._pending_available.items():
            position = self.portfolio.get_position(symbol)
            if position is not None:
                position.available_quantity += qty
        self._pending_available.clear()

    def get_snapshot(self) -> dict:
        """获取当前组合快照

        Returns:
            dict with keys: cash, total_value, market_value, positions, return_pct
        """
        positions_info = {}
        for symbol, pos in self.portfolio.positions.items():
            positions_info[symbol] = {
                "quantity": pos.quantity,
                "available_quantity": pos.available_quantity,
                "avg_cost": round(pos.avg_cost, 4),
                "current_price": round(pos.current_price, 4),
                "market_value": round(pos.market_value, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(pos.unrealized_pnl_pct, 4),
            }

        return {
            "cash": round(self.portfolio.cash, 2),
            "total_value": round(self.portfolio.total_value, 2),
            "market_value": round(self.portfolio.market_value, 2),
            "positions": positions_info,
            "return_pct": round(self.portfolio.total_return, 4),
        }
