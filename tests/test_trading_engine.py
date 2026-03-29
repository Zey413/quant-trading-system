"""交易引擎模块测试

测试覆盖：
- OrderManager: 订单创建、取消、修改、状态流转、事件回调、查询
- PositionTracker: 买入、卖出、盈亏计算、T+1规则、组合价值
- ExecutionEngine: 信号执行、买卖便捷方法、引擎控制
- SimulatedExecutor: 市价/限价撮合、滑点、佣金、手数取整
- RiskMonitor: 风控规则检查、告警、持仓限额、日亏损
- TradeLogger: 日志记录、每日汇总、CSV导出
"""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from quant_trading.core.enums import OrderSide, OrderType
from quant_trading.core.models import Signal
from quant_trading.trading.execution_engine import (
    ExecutionConfig,
    ExecutionEngine,
    ExecutionResult,
    SimulatedExecutor,
    LiveExecutor,
)
from quant_trading.trading.order_manager import (
    ManagedOrder,
    OrderCallback,
    OrderError,
    OrderEvent,
    OrderEventType,
    OrderManager,
    OrderNotFoundError,
    OrderStateError,
    OrderStatus,
)
from quant_trading.trading.position_tracker import (
    PnLReport,
    PositionTracker,
    TrackedPosition,
)
from quant_trading.trading.risk_monitor import (
    AlertLevel,
    ConcentrationRule,
    DailyLossLimitRule,
    MaxDrawdownRule,
    MaxPositionRule,
    RiskAlert,
    RiskCheckResult,
    RiskMonitor,
    RiskRule,
)
from quant_trading.trading.trade_logger import (
    DailySummary,
    TradeLogEntry,
    TradeLogger,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def order_manager() -> OrderManager:
    return OrderManager()


@pytest.fixture
def position_tracker() -> PositionTracker:
    return PositionTracker(initial_capital=1_000_000.0)


@pytest.fixture
def sim_executor() -> SimulatedExecutor:
    config = ExecutionConfig(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax_rate=0.001,
        slippage=0.001,
        lot_size=100,
    )
    return SimulatedExecutor(config)


@pytest.fixture
def engine(position_tracker: PositionTracker) -> ExecutionEngine:
    config = ExecutionConfig(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax_rate=0.001,
        slippage=0.001,
        lot_size=100,
    )
    engine = ExecutionEngine(
        position_tracker=position_tracker,
        config=config,
    )
    engine.start()
    return engine


@pytest.fixture
def risk_monitor() -> RiskMonitor:
    return RiskMonitor.create_default(
        max_position_pct=0.3,
        max_drawdown=0.2,
        max_daily_loss=50000.0,
        max_concentration=0.8,
    )


@pytest.fixture
def trade_logger() -> TradeLogger:
    return TradeLogger()


def make_signal(
    symbol: str = "000001",
    signal_type: str = "buy",
    price: float = 10.0,
    strategy_name: str = "test_strategy",
) -> Signal:
    return Signal(
        date=date.today(),
        symbol=symbol,
        signal_type=signal_type,
        price=price,
        strategy_name=strategy_name,
    )


# ============================================================
# OrderManager 测试
# ============================================================


class TestOrderManager:
    """OrderManager 测试"""

    def test_submit_order(self, order_manager: OrderManager) -> None:
        """测试提交订单"""
        order = order_manager.submit_order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        assert order.symbol == "000001"
        assert order.side == OrderSide.BUY
        assert order.quantity == 1000
        assert order.status == OrderStatus.PENDING
        assert order_manager.order_count == 1

    def test_submit_order_invalid_quantity(
        self, order_manager: OrderManager
    ) -> None:
        """测试提交数量为0的订单"""
        with pytest.raises(OrderError, match="数量必须大于0"):
            order_manager.submit_order(
                symbol="000001",
                side=OrderSide.BUY,
                quantity=0,
                price=10.0,
            )

    def test_submit_limit_order_no_price(
        self, order_manager: OrderManager
    ) -> None:
        """测试限价单价格为0"""
        with pytest.raises(OrderError, match="限价单价格必须大于0"):
            order_manager.submit_order(
                symbol="000001",
                side=OrderSide.BUY,
                quantity=100,
                price=0.0,
                order_type=OrderType.LIMIT,
            )

    def test_cancel_order(self, order_manager: OrderManager) -> None:
        """测试取消订单"""
        order = order_manager.submit_order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        cancelled = order_manager.cancel_order(order.order_id, reason="测试取消")
        assert cancelled.status == OrderStatus.CANCELLED

    def test_cancel_filled_order_fails(
        self, order_manager: OrderManager
    ) -> None:
        """测试取消已成交订单应失败"""
        order = order_manager.submit_order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        order_manager.update_order_status(
            order.order_id, OrderStatus.SUBMITTED
        )
        order_manager.update_order_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_quantity=1000,
            filled_price=10.0,
        )
        with pytest.raises(OrderStateError):
            order_manager.cancel_order(order.order_id)

    def test_modify_order(self, order_manager: OrderManager) -> None:
        """测试修改订单"""
        order = order_manager.submit_order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        modified = order_manager.modify_order(
            order.order_id, quantity=2000, price=9.5
        )
        assert modified.quantity == 2000
        assert modified.price == 9.5

    def test_modify_done_order_fails(
        self, order_manager: OrderManager
    ) -> None:
        """测试修改已终结的订单应失败"""
        order = order_manager.submit_order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        order_manager.cancel_order(order.order_id)
        with pytest.raises(OrderStateError):
            order_manager.modify_order(order.order_id, quantity=500)

    def test_get_order_not_found(self, order_manager: OrderManager) -> None:
        """测试查询不存在的订单"""
        with pytest.raises(OrderNotFoundError):
            order_manager.get_order("nonexistent")

    def test_get_pending_orders(self, order_manager: OrderManager) -> None:
        """测试获取活跃订单"""
        order_manager.submit_order(
            symbol="000001", side=OrderSide.BUY, quantity=100, price=10.0
        )
        order_manager.submit_order(
            symbol="000002", side=OrderSide.BUY, quantity=200, price=20.0
        )
        o3 = order_manager.submit_order(
            symbol="000003", side=OrderSide.SELL, quantity=300, price=30.0
        )
        order_manager.cancel_order(o3.order_id)

        pending = order_manager.get_pending_orders()
        assert len(pending) == 2

    def test_get_order_history_with_filter(
        self, order_manager: OrderManager
    ) -> None:
        """测试带过滤条件的订单历史查询"""
        order_manager.submit_order(
            symbol="000001", side=OrderSide.BUY, quantity=100, price=10.0
        )
        order_manager.submit_order(
            symbol="000002", side=OrderSide.BUY, quantity=200, price=20.0
        )

        history = order_manager.get_order_history(symbol="000001")
        assert len(history) == 1
        assert history[0].symbol == "000001"

    def test_event_callback(self, order_manager: OrderManager) -> None:
        """测试事件回调系统"""
        events: list[OrderEvent] = []
        order_manager.register_callback(lambda e: events.append(e))

        order_manager.submit_order(
            symbol="000001", side=OrderSide.BUY, quantity=100, price=10.0
        )
        assert len(events) == 1
        assert events[0].event_type == OrderEventType.CREATED

    def test_unregister_callback(self, order_manager: OrderManager) -> None:
        """测试注销回调"""
        events: list[OrderEvent] = []
        callback = lambda e: events.append(e)
        order_manager.register_callback(callback)
        order_manager.unregister_callback(callback)

        order_manager.submit_order(
            symbol="000001", side=OrderSide.BUY, quantity=100, price=10.0
        )
        assert len(events) == 0

    def test_order_status_transitions(
        self, order_manager: OrderManager
    ) -> None:
        """测试订单状态流转"""
        order = order_manager.submit_order(
            symbol="000001", side=OrderSide.BUY, quantity=1000, price=10.0
        )
        # PENDING -> SUBMITTED
        order_manager.update_order_status(
            order.order_id, OrderStatus.SUBMITTED
        )
        assert order.status == OrderStatus.SUBMITTED

        # SUBMITTED -> PARTIAL_FILLED
        order_manager.update_order_status(
            order.order_id,
            OrderStatus.PARTIAL_FILLED,
            filled_quantity=500,
            filled_price=10.0,
        )
        assert order.status == OrderStatus.PARTIAL_FILLED
        assert order.filled_quantity == 500

        # PARTIAL_FILLED -> FILLED
        order_manager.update_order_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_quantity=500,
            filled_price=10.1,
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1000

    def test_invalid_status_transition(
        self, order_manager: OrderManager
    ) -> None:
        """测试非法状态转换"""
        order = order_manager.submit_order(
            symbol="000001", side=OrderSide.BUY, quantity=100, price=10.0
        )
        order_manager.cancel_order(order.order_id)
        # CANCELLED -> FILLED 是非法转换
        with pytest.raises(OrderStateError):
            order_manager.update_order_status(
                order.order_id, OrderStatus.FILLED
            )


# ============================================================
# PositionTracker 测试
# ============================================================


class TestPositionTracker:
    """PositionTracker 测试"""

    def test_buy_new_position(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试买入新建仓位"""
        result = position_tracker.update_position(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
            commission=5.0,
        )
        assert result.symbol == "000001"
        assert result.side == "buy"
        pos = position_tracker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 1000
        assert pos.avg_cost == 10.0
        assert pos.available_quantity == 0  # T+1

    def test_buy_add_position(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试加仓"""
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=10.0, commission=5.0,
        )
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=12.0, commission=5.0,
        )
        pos = position_tracker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 2000
        assert pos.avg_cost == pytest.approx(11.0)

    def test_sell_position(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试卖出减仓"""
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=10.0,
        )
        # 模拟T+1: 先释放可卖数量
        position_tracker.new_trading_day()

        result = position_tracker.update_position(
            symbol="000001", side=OrderSide.SELL,
            quantity=500, price=12.0, commission=5.0, tax=6.0,
        )
        assert result.realized_pnl == pytest.approx(
            (12.0 - 10.0) * 500 - 5.0 - 6.0
        )
        pos = position_tracker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 500

    def test_sell_clear_position(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试清仓"""
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=10.0,
        )
        position_tracker.new_trading_day()

        position_tracker.update_position(
            symbol="000001", side=OrderSide.SELL,
            quantity=1000, price=11.0,
        )
        assert position_tracker.get_position("000001") is None
        assert not position_tracker.has_position("000001")

    def test_sell_insufficient_available(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试可卖数量不足（T+1）"""
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=10.0,
        )
        # 不调用new_trading_day，available_quantity=0
        with pytest.raises(ValueError, match="可卖数量不足"):
            position_tracker.update_position(
                symbol="000001", side=OrderSide.SELL,
                quantity=1000, price=11.0,
            )

    def test_portfolio_value(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试组合总值计算"""
        initial = position_tracker.get_portfolio_value()
        assert initial == 1_000_000.0

        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=10.0, commission=5.0,
        )
        # 市值 = 1000 * 10.0 = 10000
        # 花费 = 10000 + 5.0 = 10005.0
        # 总值 = (1000000 - 10005) + 10000 = 999995.0
        expected = 1_000_000.0 - 5.0
        assert position_tracker.get_portfolio_value() == pytest.approx(
            expected, abs=0.01
        )

    def test_pnl_report(self, position_tracker: PositionTracker) -> None:
        """测试盈亏报告"""
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=10.0,
        )
        position_tracker.update_price("000001", 12.0)

        report = position_tracker.get_pnl()
        assert report.position_count == 1
        assert report.total_market_value == pytest.approx(12000.0)
        assert report.total_unrealized_pnl == pytest.approx(2000.0)

    def test_t_plus_1_rule(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试T+1规则"""
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=1000, price=10.0,
        )
        pos = position_tracker.get_position("000001")
        assert pos is not None
        assert pos.available_quantity == 0

        # 新交易日后可卖
        position_tracker.new_trading_day()
        pos = position_tracker.get_position("000001")
        assert pos is not None
        assert pos.available_quantity == 1000

    def test_update_prices_batch(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试批量价格更新"""
        position_tracker.update_position(
            symbol="000001", side=OrderSide.BUY,
            quantity=100, price=10.0,
        )
        position_tracker.update_position(
            symbol="000002", side=OrderSide.BUY,
            quantity=200, price=20.0,
        )
        position_tracker.update_prices({"000001": 11.0, "000002": 22.0})

        pos1 = position_tracker.get_position("000001")
        pos2 = position_tracker.get_position("000002")
        assert pos1 is not None and pos1.current_price == 11.0
        assert pos2 is not None and pos2.current_price == 22.0


# ============================================================
# ExecutionEngine 测试
# ============================================================


class TestExecutionEngine:
    """ExecutionEngine 测试"""

    def test_engine_start_stop(self) -> None:
        """测试引擎启停"""
        engine = ExecutionEngine()
        assert not engine.is_running

        engine.start()
        assert engine.is_running

        engine.stop()
        assert not engine.is_running

    def test_execute_signal_not_running(self) -> None:
        """测试未启动时执行信号"""
        engine = ExecutionEngine()
        signal = make_signal()
        result = engine.execute_signal(signal, market_price=10.0, quantity=100)
        assert not result.success
        assert "未启动" in result.message

    def test_execute_buy_signal(self, engine: ExecutionEngine) -> None:
        """测试执行买入信号"""
        signal = make_signal(signal_type="buy", price=10.0)
        result = engine.execute_signal(
            signal, market_price=10.0, quantity=1000
        )
        assert result.success
        assert result.filled_quantity == 1000
        assert result.filled_price > 0

    def test_execute_sell_signal(self, engine: ExecutionEngine) -> None:
        """测试执行卖出信号"""
        # 先买入
        buy_signal = make_signal(signal_type="buy", price=10.0)
        engine.execute_signal(buy_signal, market_price=10.0, quantity=1000)

        # T+1
        engine.position_tracker.new_trading_day()

        # 卖出
        sell_signal = make_signal(signal_type="sell", price=11.0)
        result = engine.execute_signal(
            sell_signal, market_price=11.0, quantity=1000
        )
        assert result.success

    def test_buy_convenience_method(self, engine: ExecutionEngine) -> None:
        """测试买入便捷方法"""
        result = engine.buy("000001", 1000, 10.0)
        assert result.success
        pos = engine.position_tracker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 1000

    def test_sell_convenience_method(self, engine: ExecutionEngine) -> None:
        """测试卖出便捷方法"""
        engine.buy("000001", 1000, 10.0)
        engine.position_tracker.new_trading_day()

        result = engine.sell("000001", 1000, 11.0)
        assert result.success

    def test_execution_history(self, engine: ExecutionEngine) -> None:
        """测试执行历史"""
        engine.buy("000001", 1000, 10.0)
        engine.buy("000002", 500, 20.0)

        history = engine.get_execution_history()
        assert len(history) == 2
        assert engine.execution_count == 2
        assert engine.success_count == 2

    def test_stop_cancels_pending_orders(
        self, engine: ExecutionEngine
    ) -> None:
        """测试停止引擎取消活跃订单"""
        # 直接在order_manager中创建一个待处理订单
        engine.order_manager.submit_order(
            symbol="000001", side=OrderSide.BUY, quantity=100, price=10.0
        )
        assert engine.order_manager.active_order_count == 1

        engine.stop()
        assert engine.order_manager.active_order_count == 0


# ============================================================
# SimulatedExecutor 测试
# ============================================================


class TestSimulatedExecutor:
    """SimulatedExecutor 测试"""

    def test_market_order_execution(
        self, sim_executor: SimulatedExecutor
    ) -> None:
        """测试市价单撮合"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
            order_type=OrderType.MARKET,
        )
        result = sim_executor.execute(order, market_price=10.0)
        assert result.success
        assert result.filled_quantity == 1000
        # 买入有正滑点
        assert result.filled_price > 10.0

    def test_limit_order_buy_can_fill(
        self, sim_executor: SimulatedExecutor
    ) -> None:
        """测试限价买入 - 市价低于限价可成交"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.5,
            order_type=OrderType.LIMIT,
        )
        result = sim_executor.execute(order, market_price=10.0)
        assert result.success

    def test_limit_order_buy_cannot_fill(
        self, sim_executor: SimulatedExecutor
    ) -> None:
        """测试限价买入 - 市价高于限价不成交"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=9.0,
            order_type=OrderType.LIMIT,
        )
        result = sim_executor.execute(order, market_price=10.0)
        assert not result.success

    def test_sell_with_stamp_tax(
        self, sim_executor: SimulatedExecutor
    ) -> None:
        """测试卖出收取印花税"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=1000,
            price=10.0,
            order_type=OrderType.MARKET,
        )
        result = sim_executor.execute(order, market_price=10.0)
        assert result.success
        assert result.tax > 0  # 卖出有印花税
        assert result.commission > 0

    def test_lot_size_rounding(
        self, sim_executor: SimulatedExecutor
    ) -> None:
        """测试手数取整"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=150,  # 不是100的整数倍
            price=10.0,
            order_type=OrderType.MARKET,
        )
        result = sim_executor.execute(order, market_price=10.0)
        assert result.success
        assert result.filled_quantity == 100  # 向下取整到100

    def test_invalid_market_price(
        self, sim_executor: SimulatedExecutor
    ) -> None:
        """测试无效市场价格"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        result = sim_executor.execute(order, market_price=0.0)
        assert not result.success


# ============================================================
# RiskMonitor 测试
# ============================================================


class TestRiskMonitor:
    """RiskMonitor 测试"""

    def test_max_position_rule(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试最大持仓比例规则"""
        rule = MaxPositionRule(max_pct=0.3)
        # 大额买入超出限制
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=50000,
            price=10.0,  # 50万 > 30%的100万
        )
        result = rule.check(order, position_tracker)
        assert not result.passed

    def test_max_position_rule_sell_always_pass(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试卖出不受持仓限制"""
        rule = MaxPositionRule(max_pct=0.3)
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=50000,
            price=10.0,
        )
        result = rule.check(order, position_tracker)
        assert result.passed

    def test_max_drawdown_rule(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试最大回撤规则"""
        rule = MaxDrawdownRule(max_drawdown=0.1)
        # 模拟亏损: 直接减少现金
        position_tracker._cash = 800_000.0  # 回撤20%
        position_tracker._peak_value = 1_000_000.0

        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        result = rule.check(order, position_tracker)
        assert not result.passed

    def test_daily_loss_limit_rule(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试日亏损限额规则"""
        rule = DailyLossLimitRule(max_daily_loss=10000.0)
        rule.record_loss(-15000.0)  # 超过限额

        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        result = rule.check(order, position_tracker)
        assert not result.passed

    def test_daily_loss_limit_reset(self) -> None:
        """测试日亏损重置"""
        rule = DailyLossLimitRule(max_daily_loss=10000.0)
        rule.record_loss(-5000.0)
        assert rule.daily_loss == 5000.0

        rule.reset_daily()
        assert rule.daily_loss == 0.0

    def test_concentration_rule(
        self, position_tracker: PositionTracker
    ) -> None:
        """测试集中度规则"""
        rule = ConcentrationRule(max_concentration=0.5)
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=60000,
            price=10.0,  # 60万 > 50%的100万
        )
        result = rule.check(order, position_tracker)
        assert not result.passed

    def test_check_pre_trade(
        self,
        risk_monitor: RiskMonitor,
        position_tracker: PositionTracker,
    ) -> None:
        """测试交易前风控检查 - 全部通过"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,  # 1万 << 30万限制
        )
        passed, results = risk_monitor.check_pre_trade(order, position_tracker)
        assert passed
        assert len(results) == 4  # 4条默认规则

    def test_check_pre_trade_fail(
        self,
        risk_monitor: RiskMonitor,
        position_tracker: PositionTracker,
    ) -> None:
        """测试交易前风控检查 - 未通过"""
        # 买入超过集中度限制
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=100000,
            price=10.0,  # 100万 > 80%限制
        )
        passed, results = risk_monitor.check_pre_trade(order, position_tracker)
        assert not passed

    def test_alert_callback(self, risk_monitor: RiskMonitor) -> None:
        """测试告警回调"""
        alerts: list[RiskAlert] = []
        risk_monitor.register_alert_callback(lambda a: alerts.append(a))

        risk_monitor.alert(
            AlertLevel.WARNING, "test_rule", "测试告警"
        )
        assert len(alerts) == 1
        assert alerts[0].message == "测试告警"

    def test_add_remove_rule(self) -> None:
        """测试添加/移除规则"""
        monitor = RiskMonitor()
        rule = MaxPositionRule(max_pct=0.3)
        monitor.add_rule(rule)
        assert len(monitor.get_rules()) == 1

        monitor.remove_rule("max_position_rule")
        assert len(monitor.get_rules()) == 0

    def test_check_daily_loss_limit_method(
        self, risk_monitor: RiskMonitor
    ) -> None:
        """测试日亏损限额检查方法"""
        result = risk_monitor.check_daily_loss_limit(
            daily_pnl=-60000.0, max_daily_loss=50000.0
        )
        assert not result.passed

    def test_check_position_limit_method(
        self,
        risk_monitor: RiskMonitor,
        position_tracker: PositionTracker,
    ) -> None:
        """测试持仓限额检查方法"""
        result = risk_monitor.check_position_limit(
            position_tracker, max_total_pct=0.8
        )
        assert result.passed  # 无持仓应该通过


# ============================================================
# TradeLogger 测试
# ============================================================


class TestTradeLogger:
    """TradeLogger 测试"""

    def test_log_order(self, trade_logger: TradeLogger) -> None:
        """测试记录订单"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        trade_logger.log_order(order, "创建订单")
        assert trade_logger.order_count == 1

    def test_log_trade(self, trade_logger: TradeLogger) -> None:
        """测试记录成交"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        trade_logger.log_trade(
            order, filled_price=10.01, filled_quantity=1000,
            commission=5.0, pnl=0.0,
        )
        assert trade_logger.trade_count == 1

    def test_log_risk_event(self, trade_logger: TradeLogger) -> None:
        """测试记录风控事件"""
        trade_logger.log_risk_event(
            level="warning",
            rule_name="test_rule",
            message="测试风控事件",
        )
        assert trade_logger.risk_event_count == 1

    def test_log_risk_event_from_alert(
        self, trade_logger: TradeLogger
    ) -> None:
        """测试从RiskAlert记录风控事件"""
        alert = RiskAlert(
            level=AlertLevel.CRITICAL,
            rule_name="max_drawdown",
            message="回撤超限",
        )
        trade_logger.log_risk_event(alert=alert)
        logs = trade_logger.get_risk_logs()
        assert len(logs) == 1
        assert logs[0].level == "critical"

    def test_get_daily_summary(self, trade_logger: TradeLogger) -> None:
        """测试每日汇总"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        trade_logger.log_trade(
            order, filled_price=10.0, filled_quantity=1000,
            commission=5.0, pnl=0.0,
        )

        summary = trade_logger.get_daily_summary(date.today())
        assert summary.total_trades == 1
        assert summary.buy_trades == 1
        assert summary.total_volume == 1000

    def test_export_trades_csv(self, trade_logger: TradeLogger) -> None:
        """测试导出交易记录到CSV"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        trade_logger.log_trade(
            order, filled_price=10.01, filled_quantity=1000,
            commission=5.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trades.csv"
            count = trade_logger.export_trades(filepath)
            assert count == 1
            assert filepath.exists()

            # 验证CSV内容
            with open(filepath, encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 2  # header + 1 row

    def test_export_empty(self, trade_logger: TradeLogger) -> None:
        """测试导出空记录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trades.csv"
            count = trade_logger.export_trades(filepath)
            assert count == 0

    def test_on_order_event_callback(
        self, trade_logger: TradeLogger
    ) -> None:
        """测试作为OrderManager回调"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        event = OrderEvent(
            event_type=OrderEventType.CREATED,
            order=order,
        )
        trade_logger.on_order_event(event)
        assert trade_logger.order_count == 1

    def test_clear(self, trade_logger: TradeLogger) -> None:
        """测试清空日志"""
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        trade_logger.log_order(order)
        trade_logger.log_risk_event(message="test")
        trade_logger.clear()
        assert trade_logger.order_count == 0
        assert trade_logger.risk_event_count == 0


# ============================================================
# LiveExecutor 测试
# ============================================================


class TestLiveExecutor:
    """LiveExecutor 测试 - 抽象接口验证"""

    def test_not_implemented(self) -> None:
        """测试未实现的方法抛出NotImplementedError"""
        executor = LiveExecutor(broker_name="测试券商")
        order = ManagedOrder(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
        )
        tracker = PositionTracker()

        with pytest.raises(NotImplementedError):
            executor.execute(order, 10.0)

        with pytest.raises(NotImplementedError):
            executor.validate_order(order, tracker)

        with pytest.raises(NotImplementedError):
            executor.connect()

        with pytest.raises(NotImplementedError):
            executor.disconnect()

        with pytest.raises(NotImplementedError):
            executor.query_position()

        with pytest.raises(NotImplementedError):
            executor.query_balance()
