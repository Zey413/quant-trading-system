"""执行引擎 - 信号到订单的执行链路

负责：
- 接收策略信号，转化为订单执行
- SimulatedExecutor: 模拟撮合（限价/市价单, 滑点, 手续费, T+1）
- LiveExecutor: 抽象接口预留实盘券商对接
- 引擎启停控制与线程安全
"""

from __future__ import annotations

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from quant_trading.core.config import BrokerConfig
from quant_trading.core.enums import OrderSide, OrderType
from quant_trading.core.models import Signal
from quant_trading.trading.order_manager import (
    ManagedOrder,
    OrderManager,
    OrderStatus,
)
from quant_trading.trading.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


# ============================================================
# 执行结果模型
# ============================================================


class ExecutionResult(BaseModel):
    """执行结果

    Attributes:
        success: 是否成功
        order: 执行的订单
        message: 结果说明
        filled_price: 成交价格
        filled_quantity: 成交数量
        commission: 佣金
        tax: 印花税
        slippage_cost: 滑点成本
    """
    success: bool = False
    order: Optional[ManagedOrder] = None
    message: str = ""
    filled_price: float = 0.0
    filled_quantity: int = 0
    commission: float = 0.0
    tax: float = 0.0
    slippage_cost: float = 0.0


class ExecutionConfig(BaseModel):
    """执行引擎配置"""
    commission_rate: float = Field(default=0.0003, description="佣金率")
    min_commission: float = Field(default=5.0, description="最低佣金")
    stamp_tax_rate: float = Field(default=0.001, description="印花税率(仅卖出)")
    slippage: float = Field(default=0.001, description="滑点比例")
    lot_size: int = Field(default=100, description="最小交易单位")
    t_plus_1: bool = Field(default=True, description="T+1规则")

    @classmethod
    def from_broker_config(cls, broker_config: BrokerConfig) -> ExecutionConfig:
        """从券商配置创建

        Args:
            broker_config: 券商配置

        Returns:
            ExecutionConfig
        """
        return cls(
            commission_rate=broker_config.commission_rate,
            min_commission=broker_config.min_commission,
            stamp_tax_rate=broker_config.stamp_tax_rate,
            slippage=broker_config.slippage,
            lot_size=broker_config.lot_size,
            t_plus_1=broker_config.t_plus_1,
        )


# ============================================================
# 执行器抽象基类
# ============================================================


class BaseExecutor(ABC):
    """执行器抽象基类

    定义撮合接口，由具体实现（模拟/实盘）完成撮合逻辑。
    """

    @abstractmethod
    def execute(
        self,
        order: ManagedOrder,
        market_price: float,
        current_date: Optional[date] = None,
    ) -> ExecutionResult:
        """执行订单

        Args:
            order: 待执行的订单
            market_price: 当前市场价格
            current_date: 当前交易日期

        Returns:
            ExecutionResult: 执行结果
        """
        ...

    @abstractmethod
    def validate_order(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> tuple[bool, str]:
        """验证订单

        Args:
            order: 待验证的订单
            position_tracker: 持仓跟踪器

        Returns:
            (is_valid, reason)
        """
        ...


# ============================================================
# SimulatedExecutor - 模拟撮合
# ============================================================


class SimulatedExecutor(BaseExecutor):
    """模拟撮合执行器

    模拟A股交易规则，包括：
    - 市价单 / 限价单撮合
    - 滑点模拟
    - 佣金与印花税计算
    - 最小交易单位 (手)
    - T+1规则检查

    Attributes:
        config: 执行配置
    """

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def execute(
        self,
        order: ManagedOrder,
        market_price: float,
        current_date: Optional[date] = None,
    ) -> ExecutionResult:
        """模拟撮合订单

        Args:
            order: 待执行订单
            market_price: 当前市场价格
            current_date: 当前交易日期

        Returns:
            ExecutionResult: 撮合结果
        """
        if market_price <= 0:
            return ExecutionResult(
                success=False,
                order=order,
                message=f"市场价格无效: {market_price}",
            )

        # 数量取整到 lot_size
        quantity = self._round_to_lot_size(order.quantity)
        if quantity <= 0:
            return ExecutionResult(
                success=False,
                order=order,
                message=f"数量不足一手({self.config.lot_size}股)",
            )

        # 限价单: 检查价格是否可成交
        if order.order_type == OrderType.LIMIT:
            if not self._can_fill_limit_order(order, market_price):
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=(
                        f"限价单未成交: "
                        f"{'买入限价' if order.side == OrderSide.BUY else '卖出限价'}"
                        f" {order.price:.4f}, 市价 {market_price:.4f}"
                    ),
                )

        # 计算成交价（含滑点）
        if order.side == OrderSide.BUY:
            filled_price = market_price * (1 + self.config.slippage)
        else:
            filled_price = market_price * (1 - self.config.slippage)

        filled_price = max(filled_price, 0.01)
        slippage_cost = abs(filled_price - market_price) * quantity

        # 计算费用
        turnover = filled_price * quantity
        commission = max(
            turnover * self.config.commission_rate,
            self.config.min_commission,
        )
        tax = 0.0
        if order.side == OrderSide.SELL:
            tax = turnover * self.config.stamp_tax_rate

        return ExecutionResult(
            success=True,
            order=order,
            message="模拟成交",
            filled_price=round(filled_price, 4),
            filled_quantity=quantity,
            commission=round(commission, 2),
            tax=round(tax, 2),
            slippage_cost=round(slippage_cost, 2),
        )

    def validate_order(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> tuple[bool, str]:
        """验证订单

        买入: 检查资金是否充足
        卖出: 检查可卖数量是否充足

        Args:
            order: 待验证的订单
            position_tracker: 持仓跟踪器

        Returns:
            (is_valid, reason)
        """
        quantity = self._round_to_lot_size(order.quantity)
        if quantity <= 0:
            return False, f"数量不足一手({self.config.lot_size}股)"

        if order.side == OrderSide.BUY:
            return self._validate_buy(order, position_tracker, quantity)
        else:
            return self._validate_sell(order, position_tracker, quantity)

    def _validate_buy(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
        quantity: int,
    ) -> tuple[bool, str]:
        """验证买入订单"""
        price = order.price if order.price > 0 else 10.0  # 市价单估算
        estimated_price = price * (1 + self.config.slippage)
        estimated_turnover = estimated_price * quantity
        estimated_commission = max(
            estimated_turnover * self.config.commission_rate,
            self.config.min_commission,
        )
        total_cost = estimated_turnover + estimated_commission

        if total_cost > position_tracker.cash:
            return (
                False,
                f"资金不足: 需要 {total_cost:.2f}，可用 {position_tracker.cash:.2f}",
            )
        return True, "验证通过"

    def _validate_sell(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
        quantity: int,
    ) -> tuple[bool, str]:
        """验证卖出订单"""
        position = position_tracker.get_position(order.symbol)
        if position is None or position.quantity <= 0:
            return False, f"未持有 {order.symbol}"

        available = position.available_quantity
        if quantity > available:
            return (
                False,
                f"可卖数量不足: 需要 {quantity}，可卖 {available}"
                f"（总持仓 {position.quantity}）",
            )
        return True, "验证通过"

    def _can_fill_limit_order(
        self,
        order: ManagedOrder,
        market_price: float,
    ) -> bool:
        """限价单是否可以成交"""
        if order.side == OrderSide.BUY:
            # 买入限价单: 市价 <= 限价 才能成交
            return market_price <= order.price
        else:
            # 卖出限价单: 市价 >= 限价 才能成交
            return market_price >= order.price

    def _round_to_lot_size(self, quantity: int) -> int:
        """向下取整到 lot_size 整数倍"""
        return (quantity // self.config.lot_size) * self.config.lot_size


# ============================================================
# LiveExecutor - 实盘执行器（抽象接口）
# ============================================================


class LiveExecutor(BaseExecutor):
    """实盘执行器 - 抽象接口

    预留券商实盘对接接口，子类需实现具体的券商API调用。
    可对接的券商包括: 华泰证券(easytrader)、东方财富、同花顺等。

    用法::

        class HuataiExecutor(LiveExecutor):
            def __init__(self, account, password):
                super().__init__(broker_name="华泰证券")
                self._client = easytrader.use('ht')
                self._client.connect(account, password)

            def execute(self, order, market_price, current_date=None):
                # 调用券商API下单
                ...

    Attributes:
        broker_name: 券商名称
        is_connected: 是否已连接
    """

    def __init__(self, broker_name: str = "unknown") -> None:
        self.broker_name = broker_name
        self.is_connected = False

    def connect(self, **kwargs: Any) -> bool:
        """连接券商

        Args:
            **kwargs: 连接参数（账号、密码等）

        Returns:
            是否连接成功
        """
        raise NotImplementedError(
            f"请在子类中实现 {self.broker_name} 的连接逻辑"
        )

    def disconnect(self) -> None:
        """断开连接"""
        raise NotImplementedError(
            f"请在子类中实现 {self.broker_name} 的断开逻辑"
        )

    def execute(
        self,
        order: ManagedOrder,
        market_price: float,
        current_date: Optional[date] = None,
    ) -> ExecutionResult:
        """实盘下单 - 需子类实现"""
        raise NotImplementedError(
            f"请在子类中实现 {self.broker_name} 的下单逻辑"
        )

    def validate_order(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> tuple[bool, str]:
        """验证订单 - 需子类实现"""
        raise NotImplementedError(
            f"请在子类中实现 {self.broker_name} 的订单验证逻辑"
        )

    def query_position(self) -> dict[str, Any]:
        """查询实盘持仓 - 需子类实现"""
        raise NotImplementedError(
            f"请在子类中实现 {self.broker_name} 的持仓查询逻辑"
        )

    def query_balance(self) -> dict[str, float]:
        """查询资金余额 - 需子类实现"""
        raise NotImplementedError(
            f"请在子类中实现 {self.broker_name} 的资金查询逻辑"
        )


# ============================================================
# ExecutionEngine - 执行引擎
# ============================================================


class ExecutionEngine:
    """执行引擎

    协调 OrderManager、PositionTracker 和 Executor，
    实现从策略信号到订单成交的完整链路。

    Attributes:
        order_manager: 订单管理器
        position_tracker: 持仓跟踪器
        executor: 撮合执行器
        _running: 引擎是否运行中
        _lock: 线程锁
    """

    def __init__(
        self,
        order_manager: OrderManager | None = None,
        position_tracker: PositionTracker | None = None,
        executor: BaseExecutor | None = None,
        config: ExecutionConfig | None = None,
    ) -> None:
        """初始化执行引擎

        Args:
            order_manager: 订单管理器（None则自动创建）
            position_tracker: 持仓跟踪器（None则自动创建）
            executor: 撮合执行器（None则使用模拟执行器）
            config: 执行配置
        """
        self.order_manager = order_manager or OrderManager()
        self.position_tracker = position_tracker or PositionTracker()
        self.executor = executor or SimulatedExecutor(config)
        self._running = False
        self._lock = threading.RLock()
        self._execution_history: list[ExecutionResult] = []

    # ----------------------------------------------------------
    # 引擎控制
    # ----------------------------------------------------------

    def start(self) -> None:
        """启动引擎"""
        with self._lock:
            if self._running:
                logger.warning("执行引擎已在运行中")
                return
            self._running = True
        logger.info("执行引擎已启动")

    def stop(self) -> None:
        """停止引擎

        取消所有活跃订单后停止引擎。
        """
        with self._lock:
            if not self._running:
                logger.warning("执行引擎未在运行")
                return

            # 取消所有活跃订单
            pending_orders = self.order_manager.get_pending_orders()
            for order in pending_orders:
                try:
                    self.order_manager.cancel_order(
                        order.order_id, reason="引擎停止，自动取消"
                    )
                except Exception:
                    logger.exception(
                        "取消订单失败: %s", order.order_id
                    )

            self._running = False
        logger.info("执行引擎已停止，取消了 %d 个活跃订单", len(pending_orders))

    @property
    def is_running(self) -> bool:
        """引擎是否运行中"""
        return self._running

    # ----------------------------------------------------------
    # 信号执行
    # ----------------------------------------------------------

    def execute_signal(
        self,
        signal: Signal,
        market_price: float | None = None,
        quantity: int | None = None,
        current_date: date | None = None,
    ) -> ExecutionResult:
        """执行交易信号

        将策略信号转化为订单并执行，完整流程：
        1. 检查引擎状态
        2. 创建订单
        3. 验证订单
        4. 撮合执行
        5. 更新持仓

        Args:
            signal: 交易信号
            market_price: 市场价格（None则使用信号价格）
            quantity: 委托数量（None则自动计算）
            current_date: 交易日期

        Returns:
            ExecutionResult: 执行结果
        """
        if not self._running:
            return ExecutionResult(
                success=False,
                message="执行引擎未启动",
            )

        price = market_price or signal.price
        if price <= 0:
            return ExecutionResult(
                success=False,
                message=f"价格无效: {price}",
            )

        # 确定方向
        if signal.is_buy():
            side = OrderSide.BUY
        elif signal.is_sell():
            side = OrderSide.SELL
        else:
            return ExecutionResult(
                success=False,
                message=f"信号类型无效: {signal.signal_type}",
            )

        # 确定数量
        if quantity is None or quantity <= 0:
            quantity = self._calc_default_quantity(side, signal.symbol, price)
            if quantity <= 0:
                return ExecutionResult(
                    success=False,
                    message="无法计算有效的下单数量",
                )

        # 1. 创建订单
        try:
            order = self.order_manager.submit_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type=OrderType.MARKET,
                strategy_name=signal.strategy_name,
                reason=signal.reason,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"创建订单失败: {e}",
            )

        # 2. 验证订单
        is_valid, reason = self.executor.validate_order(
            order, self.position_tracker
        )
        if not is_valid:
            self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.REJECTED,
                reject_reason=reason,
            )
            return ExecutionResult(
                success=False,
                order=order,
                message=f"订单验证失败: {reason}",
            )

        # 3. 提交订单
        self.order_manager.update_order_status(
            order.order_id, OrderStatus.SUBMITTED
        )

        # 4. 撮合执行
        result = self.executor.execute(order, price, current_date)

        if result.success:
            # 5. 更新订单状态
            self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.FILLED,
                filled_quantity=result.filled_quantity,
                filled_price=result.filled_price,
                commission=result.commission,
                tax=result.tax,
            )

            # 6. 更新持仓
            self.position_tracker.update_position(
                symbol=signal.symbol,
                side=side,
                quantity=result.filled_quantity,
                price=result.filled_price,
                commission=result.commission,
                tax=result.tax,
                trade_date=current_date,
            )
        else:
            # 撮合失败
            self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.REJECTED,
                reject_reason=result.message,
            )

        # 记录执行历史
        with self._lock:
            self._execution_history.append(result)

        return result

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        strategy_name: str = "",
        current_date: date | None = None,
    ) -> ExecutionResult:
        """买入

        Args:
            symbol: 股票代码
            quantity: 买入数量
            price: 价格
            strategy_name: 策略名称
            current_date: 交易日期

        Returns:
            ExecutionResult
        """
        signal = Signal(
            date=current_date or date.today(),
            symbol=symbol,
            signal_type="buy",
            price=price,
            strategy_name=strategy_name,
        )
        return self.execute_signal(
            signal, market_price=price, quantity=quantity,
            current_date=current_date,
        )

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        strategy_name: str = "",
        current_date: date | None = None,
    ) -> ExecutionResult:
        """卖出

        Args:
            symbol: 股票代码
            quantity: 卖出数量
            price: 价格
            strategy_name: 策略名称
            current_date: 交易日期

        Returns:
            ExecutionResult
        """
        signal = Signal(
            date=current_date or date.today(),
            symbol=symbol,
            signal_type="sell",
            price=price,
            strategy_name=strategy_name,
        )
        return self.execute_signal(
            signal, market_price=price, quantity=quantity,
            current_date=current_date,
        )

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_execution_history(self, limit: int = 100) -> list[ExecutionResult]:
        """获取执行历史

        Args:
            limit: 返回数量限制

        Returns:
            执行结果列表（最新在前）
        """
        with self._lock:
            return list(reversed(self._execution_history[-limit:]))

    @property
    def execution_count(self) -> int:
        """总执行次数"""
        with self._lock:
            return len(self._execution_history)

    @property
    def success_count(self) -> int:
        """成功执行次数"""
        with self._lock:
            return sum(1 for r in self._execution_history if r.success)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _calc_default_quantity(
        self,
        side: OrderSide,
        symbol: str,
        price: float,
    ) -> int:
        """计算默认下单数量

        买入: 使用可用资金的10%
        卖出: 使用全部可卖数量

        Args:
            side: 方向
            symbol: 股票代码
            price: 价格

        Returns:
            取整到 lot_size 的数量
        """
        lot_size = 100  # 默认最小交易单位
        if isinstance(self.executor, SimulatedExecutor):
            lot_size = self.executor.config.lot_size

        if side == OrderSide.BUY:
            available_cash = self.position_tracker.cash
            target_value = available_cash * 0.1  # 默认10%仓位
            quantity = int(target_value / price)
        else:
            position = self.position_tracker.get_position(symbol)
            if position is None:
                return 0
            quantity = position.available_quantity

        # 取整到 lot_size
        return (quantity // lot_size) * lot_size
