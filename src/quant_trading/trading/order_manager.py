"""订单管理器 - 订单生命周期管理

负责：
- 订单创建、提交、取消、修改
- 订单状态流转 (PENDING -> SUBMITTED -> PARTIAL_FILLED/FILLED/CANCELLED/REJECTED)
- 事件回调通知
- 订单历史查询
- 线程安全的并发访问
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from quant_trading.core.enums import OrderSide, OrderType

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与模型
# ============================================================


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"                 # 待提交
    SUBMITTED = "submitted"             # 已提交
    PARTIAL_FILLED = "partial_filled"   # 部分成交
    FILLED = "filled"                   # 全部成交
    CANCELLED = "cancelled"             # 已取消
    REJECTED = "rejected"               # 已拒绝


class OrderEventType(str, Enum):
    """订单事件类型"""
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    MODIFIED = "modified"


class ManagedOrder(BaseModel):
    """受管理的订单 - Pydantic模型

    在核心 Order dataclass 之上增加了更丰富的生命周期信息。

    Attributes:
        order_id: 唯一订单ID
        symbol: 股票代码
        side: 买卖方向
        order_type: 订单类型 (market/limit)
        quantity: 委托数量
        price: 委托价格 (市价单为0)
        filled_quantity: 已成交数量
        filled_price: 成交均价
        commission: 佣金
        tax: 印花税
        status: 订单状态
        created_at: 创建时间
        updated_at: 最后更新时间
        strategy_name: 策略名称
        reason: 订单原因/备注
        reject_reason: 拒绝原因
    """
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: float = 0.0
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    strategy_name: str = ""
    reason: str = ""
    reject_reason: str = ""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_active(self) -> bool:
        """订单是否处于活跃状态 (可被取消/修改)"""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILLED,
        )

    @property
    def is_done(self) -> bool:
        """订单是否已终结"""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )

    @property
    def total_cost(self) -> float:
        """总成本 = 成交额 + 佣金 + 税"""
        return self.filled_price * self.filled_quantity + self.commission + self.tax

    @property
    def remaining_quantity(self) -> int:
        """剩余未成交数量"""
        return self.quantity - self.filled_quantity


class OrderEvent(BaseModel):
    """订单事件"""
    event_type: OrderEventType
    order: ManagedOrder
    timestamp: datetime = Field(default_factory=datetime.now)
    data: dict[str, Any] = Field(default_factory=dict)


# 回调类型
OrderCallback = Callable[[OrderEvent], None]


# ============================================================
# 异常
# ============================================================


class OrderError(Exception):
    """订单操作异常"""
    pass


class OrderNotFoundError(OrderError):
    """订单不存在"""
    pass


class OrderStateError(OrderError):
    """订单状态不允许此操作"""
    pass


# ============================================================
# 状态流转规则
# ============================================================

# 允许的状态转换
_VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {
        OrderStatus.SUBMITTED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.PARTIAL_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIAL_FILLED: {
        OrderStatus.PARTIAL_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
}


# ============================================================
# OrderManager
# ============================================================


class OrderManager:
    """订单管理器

    线程安全的订单生命周期管理器，支持事件驱动的回调通知。

    Attributes:
        _orders: 所有订单字典 {order_id: ManagedOrder}
        _callbacks: 事件回调列表
        _lock: 线程锁
    """

    def __init__(self) -> None:
        self._orders: dict[str, ManagedOrder] = {}
        self._callbacks: list[OrderCallback] = []
        self._lock = threading.RLock()

    # ----------------------------------------------------------
    # 回调管理
    # ----------------------------------------------------------

    def register_callback(self, callback: OrderCallback) -> None:
        """注册订单事件回调

        Args:
            callback: 回调函数，接收 OrderEvent 参数
        """
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: OrderCallback) -> None:
        """注销订单事件回调

        Args:
            callback: 要注销的回调函数
        """
        with self._lock:
            self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    def _emit_event(self, event: OrderEvent) -> None:
        """触发事件回调

        Args:
            event: 订单事件
        """
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception(
                    "订单事件回调异常: event=%s, order_id=%s",
                    event.event_type,
                    event.order.order_id,
                )

    # ----------------------------------------------------------
    # 订单操作
    # ----------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
        strategy_name: str = "",
        reason: str = "",
    ) -> ManagedOrder:
        """提交新订单

        创建订单并设置为 PENDING 状态，触发 CREATED 事件。

        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 委托数量
            price: 委托价格（市价单为0）
            order_type: 订单类型
            strategy_name: 策略名称
            reason: 下单原因

        Returns:
            ManagedOrder: 创建的订单

        Raises:
            OrderError: 订单参数不合法
        """
        if quantity <= 0:
            raise OrderError(f"订单数量必须大于0，当前: {quantity}")
        if order_type == OrderType.LIMIT and price <= 0:
            raise OrderError(f"限价单价格必须大于0，当前: {price}")

        with self._lock:
            order = ManagedOrder(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                strategy_name=strategy_name,
                reason=reason,
            )
            self._orders[order.order_id] = order

        logger.info(
            "订单创建: %s %s %s %d股 @ %.4f [%s]",
            order.order_id,
            side.value,
            symbol,
            quantity,
            price,
            order_type.value,
        )

        self._emit_event(OrderEvent(
            event_type=OrderEventType.CREATED,
            order=order,
        ))

        return order

    def cancel_order(self, order_id: str, reason: str = "") -> ManagedOrder:
        """取消订单

        Args:
            order_id: 订单ID
            reason: 取消原因

        Returns:
            更新后的订单

        Raises:
            OrderNotFoundError: 订单不存在
            OrderStateError: 订单状态不允许取消
        """
        with self._lock:
            order = self._get_order_locked(order_id)

            if not order.is_active:
                raise OrderStateError(
                    f"订单 {order_id} 状态为 {order.status.value}，无法取消"
                )

            self._transition(order, OrderStatus.CANCELLED)
            order.reason = reason or order.reason

        logger.info("订单取消: %s, 原因: %s", order_id, reason)
        self._emit_event(OrderEvent(
            event_type=OrderEventType.CANCELLED,
            order=order,
        ))

        return order

    def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
    ) -> ManagedOrder:
        """修改订单 (仅限活跃状态的订单)

        Args:
            order_id: 订单ID
            quantity: 新的委托数量（None表示不修改）
            price: 新的委托价格（None表示不修改）

        Returns:
            更新后的订单

        Raises:
            OrderNotFoundError: 订单不存在
            OrderStateError: 订单状态不允许修改
            OrderError: 修改参数不合法
        """
        with self._lock:
            order = self._get_order_locked(order_id)

            if not order.is_active:
                raise OrderStateError(
                    f"订单 {order_id} 状态为 {order.status.value}，无法修改"
                )

            if quantity is not None:
                if quantity <= 0:
                    raise OrderError(f"修改数量必须大于0，当前: {quantity}")
                if quantity < order.filled_quantity:
                    raise OrderError(
                        f"修改数量 {quantity} 不能小于已成交数量 {order.filled_quantity}"
                    )
                order.quantity = quantity

            if price is not None:
                if price <= 0:
                    raise OrderError(f"修改价格必须大于0，当前: {price}")
                order.price = price

            order.updated_at = datetime.now()

        logger.info(
            "订单修改: %s, quantity=%s, price=%s",
            order_id,
            quantity,
            price,
        )
        self._emit_event(OrderEvent(
            event_type=OrderEventType.MODIFIED,
            order=order,
            data={"new_quantity": quantity, "new_price": price},
        ))

        return order

    def update_order_status(
        self,
        order_id: str,
        new_status: OrderStatus,
        filled_quantity: int = 0,
        filled_price: float = 0.0,
        commission: float = 0.0,
        tax: float = 0.0,
        reject_reason: str = "",
    ) -> ManagedOrder:
        """更新订单状态（由执行引擎调用）

        Args:
            order_id: 订单ID
            new_status: 新状态
            filled_quantity: 本次成交数量
            filled_price: 本次成交价格
            commission: 佣金
            tax: 印花税
            reject_reason: 拒绝原因

        Returns:
            更新后的订单

        Raises:
            OrderNotFoundError: 订单不存在
            OrderStateError: 状态转换不合法
        """
        with self._lock:
            order = self._get_order_locked(order_id)
            self._transition(order, new_status)

            if filled_quantity > 0:
                # 计算加权平均成交价
                old_amount = order.filled_price * order.filled_quantity
                new_amount = filled_price * filled_quantity
                total_qty = order.filled_quantity + filled_quantity
                if total_qty > 0:
                    order.filled_price = (old_amount + new_amount) / total_qty
                order.filled_quantity = total_qty

            order.commission += commission
            order.tax += tax
            order.updated_at = datetime.now()

            if reject_reason:
                order.reject_reason = reject_reason

        # 触发对应事件
        event_type_map = {
            OrderStatus.SUBMITTED: OrderEventType.SUBMITTED,
            OrderStatus.PARTIAL_FILLED: OrderEventType.PARTIAL_FILLED,
            OrderStatus.FILLED: OrderEventType.FILLED,
            OrderStatus.CANCELLED: OrderEventType.CANCELLED,
            OrderStatus.REJECTED: OrderEventType.REJECTED,
        }
        event_type = event_type_map.get(new_status)
        if event_type:
            self._emit_event(OrderEvent(
                event_type=event_type,
                order=order,
            ))

        logger.debug(
            "订单状态更新: %s -> %s, 成交 %d股 @ %.4f",
            order_id,
            new_status.value,
            filled_quantity,
            filled_price,
        )

        return order

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_order(self, order_id: str) -> ManagedOrder:
        """获取订单

        Args:
            order_id: 订单ID

        Returns:
            ManagedOrder: 订单对象

        Raises:
            OrderNotFoundError: 订单不存在
        """
        with self._lock:
            return self._get_order_locked(order_id)

    def get_pending_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        """获取活跃订单列表

        Args:
            symbol: 股票代码过滤（None表示全部）

        Returns:
            活跃订单列表
        """
        with self._lock:
            orders = [o for o in self._orders.values() if o.is_active]
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            return sorted(orders, key=lambda o: o.created_at)

    def get_order_history(
        self,
        symbol: str | None = None,
        status: OrderStatus | None = None,
        limit: int = 100,
    ) -> list[ManagedOrder]:
        """获取订单历史

        Args:
            symbol: 股票代码过滤
            status: 状态过滤
            limit: 返回数量限制

        Returns:
            订单列表（按创建时间倒序）
        """
        with self._lock:
            orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]

        orders.sort(key=lambda o: o.created_at, reverse=True)
        return orders[:limit]

    def get_orders_by_strategy(self, strategy_name: str) -> list[ManagedOrder]:
        """按策略名称查询订单

        Args:
            strategy_name: 策略名称

        Returns:
            该策略的所有订单
        """
        with self._lock:
            return [
                o for o in self._orders.values()
                if o.strategy_name == strategy_name
            ]

    @property
    def order_count(self) -> int:
        """订单总数"""
        with self._lock:
            return len(self._orders)

    @property
    def active_order_count(self) -> int:
        """活跃订单数"""
        with self._lock:
            return sum(1 for o in self._orders.values() if o.is_active)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _get_order_locked(self, order_id: str) -> ManagedOrder:
        """获取订单（调用方需持有锁）"""
        order = self._orders.get(order_id)
        if order is None:
            raise OrderNotFoundError(f"订单不存在: {order_id}")
        return order

    def _transition(self, order: ManagedOrder, new_status: OrderStatus) -> None:
        """执行状态转换（调用方需持有锁）

        Raises:
            OrderStateError: 状态转换不合法
        """
        valid_targets = _VALID_TRANSITIONS.get(order.status, set())
        if new_status not in valid_targets:
            raise OrderStateError(
                f"订单 {order.order_id} 不允许从 {order.status.value} "
                f"转换到 {new_status.value}"
            )
        order.status = new_status
        order.updated_at = datetime.now()
