"""模拟券商模块 - A股交易规则实现

实现A股市场特有的交易规则：
- 最小交易单位（手）：100股
- 佣金计算：成交额 * 佣金率，最低5元
- 印花税：仅卖出时收取，成交额 * 千分之一
- 滑点模拟：买入价上浮、卖出价下浮
- T+1限制：当日买入次日才可卖出
"""

from __future__ import annotations

import logging
from datetime import date

from quant_trading.core.config import BrokerConfig
from quant_trading.core.enums import OrderSide, OrderStatus
from quant_trading.core.models import Order, Portfolio

logger = logging.getLogger(__name__)


class PriceLimitChecker:
    """涨跌停限制校验器

    A股涨跌停规则：
    - 普通股票（主板）: ±10%
    - ST / *ST 股票: ±5%
    - 科创板（688xxx）/ 创业板（30xxxx）: ±20%
    - 北交所（8xxxxx / 4xxxxx）: ±30%

    Attributes:
        enabled: 是否启用涨跌停校验
    """

    # 涨跌停幅度映射
    LIMIT_RATIOS: dict[str, float] = {
        "normal": 0.10,     # 主板普通股票 ±10%
        "st": 0.05,         # ST股票 ±5%
        "star": 0.20,       # 科创板 ±20%
        "gem": 0.20,        # 创业板 ±20%
        "bse": 0.30,        # 北交所 ±30%
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def get_board_type(self, symbol: str, is_st: bool = False) -> str:
        """根据股票代码判断板块类型

        Args:
            symbol: 股票代码 (如 "000001", "688001", "300750")
            is_st: 是否为ST股票

        Returns:
            板块类型字符串
        """
        if is_st:
            return "st"
        code = symbol.replace(".", "").strip()
        # 取纯数字部分
        digits = "".join(c for c in code if c.isdigit())
        if digits.startswith("688"):
            return "star"
        elif digits.startswith("30"):
            return "gem"
        elif digits.startswith("8") or digits.startswith("4"):
            return "bse"
        else:
            return "normal"

    def get_limit_ratio(self, symbol: str, is_st: bool = False) -> float:
        """获取涨跌停幅度

        Args:
            symbol: 股票代码
            is_st: 是否为ST股票

        Returns:
            涨跌停幅度（如0.10表示±10%）
        """
        board_type = self.get_board_type(symbol, is_st)
        return self.LIMIT_RATIOS.get(board_type, 0.10)

    def check(
        self,
        symbol: str,
        current_price: float,
        prev_close: float,
        is_st: bool = False,
    ) -> tuple[bool, bool, float, float]:
        """检查涨跌停状态

        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前一交易日收盘价
            is_st: 是否为ST股票

        Returns:
            (at_upper_limit, at_lower_limit, upper_price, lower_price)
        """
        if not self.enabled or prev_close <= 0:
            return False, False, 0.0, 0.0

        ratio = self.get_limit_ratio(symbol, is_st)
        upper_price = round(prev_close * (1 + ratio), 2)
        lower_price = round(prev_close * (1 - ratio), 2)

        at_upper = current_price >= upper_price
        at_lower = current_price <= lower_price

        return at_upper, at_lower, upper_price, lower_price

    def can_buy(
        self,
        symbol: str,
        current_price: float,
        prev_close: float,
        is_st: bool = False,
    ) -> tuple[bool, str]:
        """检查是否可以买入（涨停不可买入）

        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前一交易日收盘价
            is_st: 是否为ST股票

        Returns:
            (can_buy, reason)
        """
        if not self.enabled:
            return True, ""

        at_upper, _, upper_price, _ = self.check(
            symbol, current_price, prev_close, is_st
        )
        if at_upper:
            return (
                False,
                f"{symbol} 涨停（当前 {current_price:.2f} >= 涨停价 {upper_price:.2f}），不可买入",
            )
        return True, ""

    def can_sell(
        self,
        symbol: str,
        current_price: float,
        prev_close: float,
        is_st: bool = False,
    ) -> tuple[bool, str]:
        """检查是否可以卖出（跌停不可卖出）

        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前一交易日收盘价
            is_st: 是否为ST股票

        Returns:
            (can_sell, reason)
        """
        if not self.enabled:
            return True, ""

        _, at_lower, _, lower_price = self.check(
            symbol, current_price, prev_close, is_st
        )
        if at_lower:
            return (
                False,
                f"{symbol} 跌停（当前 {current_price:.2f} <= 跌停价 {lower_price:.2f}），不可卖出",
            )
        return True, ""


class SimulatedBroker:
    """模拟券商

    基于A股市场规则模拟订单执行，计算佣金、印花税和滑点。
    支持涨跌停限制校验。

    Attributes:
        config: 券商配置（佣金率、印花税率、滑点等）
        price_limit_checker: 涨跌停校验器
    """

    def __init__(
        self,
        config: BrokerConfig,
        enable_price_limit: bool = True,
    ) -> None:
        self.config = config
        self.price_limit_checker = PriceLimitChecker(enabled=enable_price_limit)
        # 前收盘价缓存: symbol -> prev_close
        self._prev_close: dict[str, float] = {}

    def set_prev_close(self, symbol: str, prev_close: float) -> None:
        """设置前收盘价（用于涨跌停校验）

        Args:
            symbol: 股票代码
            prev_close: 前一交易日收盘价
        """
        self._prev_close[symbol] = prev_close

    def execute_order(
        self,
        order: Order,
        current_price: float,
        current_date: date,
    ) -> Order:
        """执行订单

        根据A股规则计算成交价（含滑点）、佣金和印花税，更新订单状态。
        新增涨跌停限制校验。

        Args:
            order: 待执行的订单
            current_price: 当前市场价格
            current_date: 当前交易日期

        Returns:
            更新后的订单对象
        """
        if order.status != OrderStatus.PENDING:
            logger.warning(
                "订单 %s 状态为 %s，无法执行", order.order_id, order.status
            )
            return order

        # --- 涨跌停校验 ---
        prev_close = self._prev_close.get(order.symbol)
        if prev_close is not None and self.price_limit_checker.enabled:
            if order.side == OrderSide.BUY:
                can, reason = self.price_limit_checker.can_buy(
                    order.symbol, current_price, prev_close
                )
            else:
                can, reason = self.price_limit_checker.can_sell(
                    order.symbol, current_price, prev_close
                )
            if not can:
                order.status = OrderStatus.REJECTED
                logger.info("订单 %s 因涨跌停被拒绝: %s", order.order_id, reason)
                return order

        # 数量必须是lot_size的整数倍
        quantity = self._round_to_lot_size(order.quantity)
        if quantity <= 0:
            order.status = OrderStatus.REJECTED
            logger.info(
                "订单 %s 被拒绝：数量 %d 不足一手(%d股)",
                order.order_id,
                order.quantity,
                self.config.lot_size,
            )
            return order

        # 计算成交价（含滑点）
        if order.side == OrderSide.BUY:
            filled_price = current_price * (1 + self.config.slippage)
        else:
            filled_price = current_price * (1 - self.config.slippage)

        # 确保成交价为正
        filled_price = max(filled_price, 0.01)

        # 成交金额
        turnover = filled_price * quantity

        # 佣金：max(成交额 * 佣金率, 最低佣金)
        commission = max(
            turnover * self.config.commission_rate,
            self.config.min_commission,
        )

        # 印花税：仅卖出收取
        tax = 0.0
        if order.side == OrderSide.SELL:
            tax = turnover * self.config.stamp_tax_rate

        # 更新订单
        order.filled_price = round(filled_price, 4)
        order.filled_quantity = quantity
        order.commission = round(commission, 2)
        order.tax = round(tax, 2)
        order.status = OrderStatus.FILLED
        order.date = current_date

        logger.debug(
            "订单 %s 成交: %s %s %d股 @ %.4f, 佣金=%.2f, 税=%.2f",
            order.order_id,
            order.side,
            order.symbol,
            quantity,
            filled_price,
            commission,
            tax,
        )

        return order

    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """验证订单是否合法

        买入验证：
        - 数量 >= 1手（100股）
        - 资金充足（含预估佣金）

        卖出验证：
        - 持有该股票
        - 可卖数量充足（T+1规则下用available_quantity）
        - 数量 >= 1手

        Args:
            order: 待验证的订单
            portfolio: 当前组合状态

        Returns:
            (is_valid, reason): 是否合法及原因说明
        """
        quantity = self._round_to_lot_size(order.quantity)
        if quantity <= 0:
            return False, f"数量不足一手({self.config.lot_size}股)"

        if order.side == OrderSide.BUY:
            return self._validate_buy(order, portfolio, quantity)
        elif order.side == OrderSide.SELL:
            return self._validate_sell(order, portfolio, quantity)
        else:
            return False, f"未知的订单方向: {order.side}"

    def _validate_buy(
        self,
        order: Order,
        portfolio: Portfolio,
        quantity: int,
    ) -> tuple[bool, str]:
        """验证买入订单"""
        # 预估成交价（含滑点）
        estimated_price = order.price * (1 + self.config.slippage)
        # 预估成交额
        estimated_turnover = estimated_price * quantity
        # 预估佣金
        estimated_commission = max(
            estimated_turnover * self.config.commission_rate,
            self.config.min_commission,
        )
        # 买入总需资金
        total_cost = estimated_turnover + estimated_commission

        if total_cost > portfolio.cash:
            return (
                False,
                f"资金不足: 需要 {total_cost:.2f}，可用 {portfolio.cash:.2f}",
            )

        return True, "验证通过"

    def _validate_sell(
        self,
        order: Order,
        portfolio: Portfolio,
        quantity: int,
    ) -> tuple[bool, str]:
        """验证卖出订单"""
        position = portfolio.get_position(order.symbol)
        if position is None or position.quantity <= 0:
            return False, f"未持有 {order.symbol}"

        # T+1规则：使用可卖数量
        if self.config.t_plus_1:
            available = position.available_quantity
        else:
            available = position.quantity

        if quantity > available:
            return (
                False,
                f"可卖数量不足: 需要 {quantity}，可卖 {available}"
                f"（总持仓 {position.quantity}）",
            )

        return True, "验证通过"

    def _round_to_lot_size(self, quantity: int) -> int:
        """将数量向下取整到最近的lot_size整数倍

        A股交易必须以「手」为单位，1手 = 100股。

        Args:
            quantity: 原始股数

        Returns:
            取整后的股数（lot_size的整数倍）
        """
        return (quantity // self.config.lot_size) * self.config.lot_size

    def calculate_cost(
        self,
        side: str,
        price: float,
        quantity: int,
    ) -> dict[str, float]:
        """计算交易费用（不执行订单，仅做费用估算）

        Args:
            side: 买卖方向 ("buy" / "sell")
            price: 价格
            quantity: 股数

        Returns:
            dict with keys: turnover, commission, tax, total_cost
        """
        turnover = price * quantity
        commission = max(
            turnover * self.config.commission_rate,
            self.config.min_commission,
        )
        tax = turnover * self.config.stamp_tax_rate if side == "sell" else 0.0

        return {
            "turnover": round(turnover, 2),
            "commission": round(commission, 2),
            "tax": round(tax, 2),
            "total_cost": round(turnover + commission + tax, 2),
        }
