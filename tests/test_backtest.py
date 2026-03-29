"""回测模块测试

测试覆盖：
- SimulatedBroker: 佣金计算、印花税（仅卖出）、手数取整、T+1验证
- PortfolioManager: 买入/卖出流程、现金更新、盈亏计算
- BacktestEngine: 合成数据 + 模拟策略的完整回测
- PerformanceMetrics: 夏普比率、最大回撤计算
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.broker import SimulatedBroker
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceMetrics, PerformanceResult
from quant_trading.backtest.portfolio import PortfolioManager
from quant_trading.core.config import AppConfig, BacktestConfig, BrokerConfig, RiskConfig
from quant_trading.core.enums import OrderSide, OrderStatus, SignalType
from quant_trading.core.models import Order, Portfolio, Position, Signal, TradeRecord


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def broker_config() -> BrokerConfig:
    return BrokerConfig(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax_rate=0.001,
        slippage=0.001,
        lot_size=100,
        t_plus_1=True,
    )


@pytest.fixture
def broker(broker_config: BrokerConfig) -> SimulatedBroker:
    return SimulatedBroker(broker_config)


@pytest.fixture
def portfolio_manager() -> PortfolioManager:
    return PortfolioManager(initial_capital=1_000_000.0)


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        backtest=BacktestConfig(
            initial_capital=1_000_000.0,
            start_date="2024-01-01",
            end_date="2024-12-31",
            risk_free_rate=0.025,
        ),
        broker=BrokerConfig(
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_tax_rate=0.001,
            slippage=0.001,
            lot_size=100,
            t_plus_1=True,
        ),
        risk=RiskConfig(
            max_position_pct=0.3,
            max_total_position_pct=0.8,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            position_sizing_method="fixed_ratio",
            fixed_ratio=0.1,
        ),
    )


def make_buy_order(
    symbol: str = "000001",
    quantity: int = 1000,
    price: float = 10.0,
    order_id: str = "test-buy-001",
) -> Order:
    """创建一个买入订单"""
    return Order(
        order_id=order_id,
        date=date(2024, 3, 1),
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price,
    )


def make_sell_order(
    symbol: str = "000001",
    quantity: int = 1000,
    price: float = 11.0,
    order_id: str = "test-sell-001",
) -> Order:
    """创建一个卖出订单"""
    return Order(
        order_id=order_id,
        date=date(2024, 3, 2),
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=quantity,
        price=price,
    )


def make_portfolio_with_position(
    symbol: str = "000001",
    quantity: int = 1000,
    available_quantity: int = 1000,
    avg_cost: float = 10.0,
    current_price: float = 10.0,
    cash: float = 900_000.0,
) -> Portfolio:
    """创建一个带有持仓的组合"""
    portfolio = Portfolio(cash=cash, initial_capital=1_000_000.0)
    portfolio.positions[symbol] = Position(
        symbol=symbol,
        quantity=quantity,
        available_quantity=available_quantity,
        avg_cost=avg_cost,
        current_price=current_price,
        buy_date=date(2024, 3, 1),
    )
    return portfolio


# ============================================================
# SimulatedBroker Tests
# ============================================================


class TestSimulatedBroker:
    """模拟券商测试"""

    def test_buy_commission_normal(self, broker: SimulatedBroker) -> None:
        """测试买入佣金计算 - 正常金额"""
        order = make_buy_order(quantity=1000, price=50.0)
        executed = broker.execute_order(order, 50.0, date(2024, 3, 1))

        assert executed.status == OrderStatus.FILLED
        assert executed.filled_quantity == 1000

        # 成交价含滑点: 50 * 1.001 = 50.05
        expected_price = 50.0 * 1.001
        assert abs(executed.filled_price - expected_price) < 0.01

        # 佣金: 50.05 * 1000 * 0.0003 = 15.015 > 5元最低佣金
        turnover = executed.filled_price * executed.filled_quantity
        expected_commission = turnover * 0.0003
        assert expected_commission > 5.0  # 确认超过最低佣金
        assert abs(executed.commission - expected_commission) < 0.01

        # 买入无印花税
        assert executed.tax == 0.0

    def test_buy_commission_minimum(self, broker: SimulatedBroker) -> None:
        """测试买入佣金 - 触发最低佣金"""
        order = make_buy_order(quantity=100, price=5.0)
        executed = broker.execute_order(order, 5.0, date(2024, 3, 1))

        assert executed.status == OrderStatus.FILLED
        # 成交额: 5.005 * 100 = 500.5
        # 佣金: 500.5 * 0.0003 = 0.15015 < 5元
        # 应收最低佣金5元
        assert executed.commission == 5.0

    def test_sell_commission_and_tax(self, broker: SimulatedBroker) -> None:
        """测试卖出佣金和印花税"""
        order = make_sell_order(quantity=1000, price=50.0)
        executed = broker.execute_order(order, 50.0, date(2024, 3, 2))

        assert executed.status == OrderStatus.FILLED

        # 成交价含滑点: 50 * 0.999 = 49.95
        expected_price = 50.0 * 0.999
        assert abs(executed.filled_price - expected_price) < 0.01

        # 佣金
        turnover = executed.filled_price * executed.filled_quantity
        expected_commission = max(turnover * 0.0003, 5.0)
        assert abs(executed.commission - expected_commission) < 0.01

        # 印花税: 仅卖出收取
        expected_tax = turnover * 0.001
        assert abs(executed.tax - expected_tax) < 0.01
        assert executed.tax > 0

    def test_buy_no_stamp_tax(self, broker: SimulatedBroker) -> None:
        """测试买入不收印花税"""
        order = make_buy_order(quantity=1000, price=20.0)
        executed = broker.execute_order(order, 20.0, date(2024, 3, 1))
        assert executed.tax == 0.0

    def test_lot_size_rounding(self, broker: SimulatedBroker) -> None:
        """测试手数取整 - 向下取整到100的整数倍"""
        # 150股 -> 100股
        order = make_buy_order(quantity=150, price=10.0)
        executed = broker.execute_order(order, 10.0, date(2024, 3, 1))
        assert executed.filled_quantity == 100

    def test_lot_size_rounding_insufficient(self, broker: SimulatedBroker) -> None:
        """测试手数取整 - 不足一手被拒绝"""
        order = make_buy_order(quantity=50, price=10.0)
        executed = broker.execute_order(order, 10.0, date(2024, 3, 1))
        assert executed.status == OrderStatus.REJECTED

    def test_slippage_buy(self, broker: SimulatedBroker) -> None:
        """测试买入滑点 - 价格上浮"""
        order = make_buy_order(quantity=100, price=10.0)
        executed = broker.execute_order(order, 10.0, date(2024, 3, 1))
        # 买入滑点: 10 * 1.001 = 10.01
        assert executed.filled_price > 10.0
        assert abs(executed.filled_price - 10.01) < 0.001

    def test_slippage_sell(self, broker: SimulatedBroker) -> None:
        """测试卖出滑点 - 价格下浮"""
        order = make_sell_order(quantity=100, price=10.0)
        executed = broker.execute_order(order, 10.0, date(2024, 3, 2))
        # 卖出滑点: 10 * 0.999 = 9.99
        assert executed.filled_price < 10.0
        assert abs(executed.filled_price - 9.99) < 0.001

    def test_validate_buy_sufficient_cash(self, broker: SimulatedBroker) -> None:
        """测试买入验证 - 资金充足"""
        portfolio = Portfolio(cash=100_000.0, initial_capital=100_000.0)
        order = make_buy_order(quantity=100, price=10.0)
        valid, reason = broker.validate_order(order, portfolio)
        assert valid is True

    def test_validate_buy_insufficient_cash(self, broker: SimulatedBroker) -> None:
        """测试买入验证 - 资金不足"""
        portfolio = Portfolio(cash=500.0, initial_capital=1_000_000.0)
        order = make_buy_order(quantity=1000, price=10.0)
        valid, reason = broker.validate_order(order, portfolio)
        assert valid is False
        assert "资金不足" in reason

    def test_validate_sell_sufficient_available(self, broker: SimulatedBroker) -> None:
        """测试卖出验证 - 可卖数量充足"""
        portfolio = make_portfolio_with_position(
            quantity=1000, available_quantity=1000
        )
        order = make_sell_order(quantity=500, price=10.0)
        valid, reason = broker.validate_order(order, portfolio)
        assert valid is True

    def test_validate_sell_t_plus_1_blocked(self, broker: SimulatedBroker) -> None:
        """测试卖出验证 - T+1规则阻止（当日买入不可卖）"""
        portfolio = make_portfolio_with_position(
            quantity=1000, available_quantity=0  # 当日买入，可卖为0
        )
        order = make_sell_order(quantity=1000, price=10.0)
        valid, reason = broker.validate_order(order, portfolio)
        assert valid is False
        assert "可卖数量不足" in reason

    def test_validate_sell_no_position(self, broker: SimulatedBroker) -> None:
        """测试卖出验证 - 无持仓"""
        portfolio = Portfolio(cash=1_000_000.0, initial_capital=1_000_000.0)
        order = make_sell_order(quantity=1000, price=10.0)
        valid, reason = broker.validate_order(order, portfolio)
        assert valid is False
        assert "未持有" in reason


# ============================================================
# PortfolioManager Tests
# ============================================================


class TestPortfolioManager:
    """组合管理器测试"""

    def test_initial_state(self, portfolio_manager: PortfolioManager) -> None:
        """测试初始状态"""
        pm = portfolio_manager
        assert pm.portfolio.cash == 1_000_000.0
        assert pm.portfolio.total_value == 1_000_000.0
        assert len(pm.portfolio.positions) == 0

    def test_process_buy(self, portfolio_manager: PortfolioManager) -> None:
        """测试买入处理"""
        pm = portfolio_manager
        order = make_buy_order(quantity=1000, price=10.0)
        # 模拟已成交
        order.filled_price = 10.01  # 含滑点
        order.filled_quantity = 1000
        order.commission = 5.0
        order.tax = 0.0
        order.status = OrderStatus.FILLED

        pm.process_buy(order)

        # 验证现金减少
        total_cost = 10.01 * 1000 + 5.0
        assert abs(pm.portfolio.cash - (1_000_000.0 - total_cost)) < 0.01

        # 验证持仓
        pos = pm.portfolio.get_position("000001")
        assert pos is not None
        assert pos.quantity == 1000
        assert pos.available_quantity == 0  # T+1: 当日不可卖
        assert abs(pos.avg_cost - 10.01) < 0.001

    def test_process_buy_add_position(
        self, portfolio_manager: PortfolioManager
    ) -> None:
        """测试加仓"""
        pm = portfolio_manager

        # 第一次买入
        order1 = make_buy_order(quantity=1000, price=10.0, order_id="buy-001")
        order1.filled_price = 10.0
        order1.filled_quantity = 1000
        order1.commission = 5.0
        order1.tax = 0.0
        order1.status = OrderStatus.FILLED
        pm.process_buy(order1)

        # new_trading_day让第一次买入变可卖
        pm.new_trading_day()

        # 第二次买入（加仓）
        order2 = make_buy_order(quantity=500, price=12.0, order_id="buy-002")
        order2.filled_price = 12.0
        order2.filled_quantity = 500
        order2.commission = 5.0
        order2.tax = 0.0
        order2.status = OrderStatus.FILLED
        pm.process_buy(order2)

        pos = pm.portfolio.get_position("000001")
        assert pos is not None
        assert pos.quantity == 1500
        # 平均成本 = (10*1000 + 12*500) / 1500 = 10.6667
        expected_avg = (10.0 * 1000 + 12.0 * 500) / 1500
        assert abs(pos.avg_cost - expected_avg) < 0.001
        # 第一次的1000股已可卖，第二次的500股不可卖
        assert pos.available_quantity == 1000

    def test_process_sell(self, portfolio_manager: PortfolioManager) -> None:
        """测试卖出处理"""
        pm = portfolio_manager

        # 先买入
        buy_order = make_buy_order(quantity=1000, price=10.0)
        buy_order.filled_price = 10.0
        buy_order.filled_quantity = 1000
        buy_order.commission = 5.0
        buy_order.tax = 0.0
        buy_order.status = OrderStatus.FILLED
        pm.process_buy(buy_order)

        # T+1: 下一交易日
        pm.new_trading_day()

        cash_before_sell = pm.portfolio.cash

        # 卖出（涨到11元）
        sell_order = make_sell_order(quantity=1000, price=11.0)
        sell_order.filled_price = 10.99  # 含滑点
        sell_order.filled_quantity = 1000
        sell_order.commission = 5.0
        sell_order.tax = 10.99  # 卖出印花税
        sell_order.status = OrderStatus.FILLED

        trade = pm.process_sell(sell_order)

        # 验证交易记录
        assert trade.side == "sell"
        assert trade.quantity == 1000
        assert trade.price == 10.99

        # PnL = (10.99 - 10.0) * 1000 - 5.0 - 10.99 = 990 - 15.99 = 974.01
        expected_pnl = (10.99 - 10.0) * 1000 - 5.0 - 10.99
        assert abs(trade.pnl - expected_pnl) < 0.01

        # 验证持仓已清除
        assert pm.portfolio.get_position("000001") is None

        # 验证现金增加
        net_income = 10.99 * 1000 - 5.0 - 10.99
        expected_cash = cash_before_sell + net_income
        assert abs(pm.portfolio.cash - expected_cash) < 0.01

    def test_process_sell_partial(
        self, portfolio_manager: PortfolioManager
    ) -> None:
        """测试部分卖出"""
        pm = portfolio_manager

        # 买入1000股
        buy_order = make_buy_order(quantity=1000, price=10.0)
        buy_order.filled_price = 10.0
        buy_order.filled_quantity = 1000
        buy_order.commission = 5.0
        buy_order.tax = 0.0
        buy_order.status = OrderStatus.FILLED
        pm.process_buy(buy_order)
        pm.new_trading_day()

        # 卖出500股
        sell_order = make_sell_order(quantity=500, price=11.0)
        sell_order.filled_price = 11.0
        sell_order.filled_quantity = 500
        sell_order.commission = 5.0
        sell_order.tax = 5.5
        sell_order.status = OrderStatus.FILLED
        trade = pm.process_sell(sell_order)

        # 验证剩余持仓
        pos = pm.portfolio.get_position("000001")
        assert pos is not None
        assert pos.quantity == 500
        assert pos.available_quantity == 500

    def test_new_trading_day_t_plus_1(
        self, portfolio_manager: PortfolioManager
    ) -> None:
        """测试T+1规则：当日买入次日可卖"""
        pm = portfolio_manager

        buy_order = make_buy_order(quantity=1000, price=10.0)
        buy_order.filled_price = 10.0
        buy_order.filled_quantity = 1000
        buy_order.commission = 5.0
        buy_order.tax = 0.0
        buy_order.status = OrderStatus.FILLED
        pm.process_buy(buy_order)

        pos = pm.portfolio.get_position("000001")
        assert pos.available_quantity == 0  # 当日不可卖

        pm.new_trading_day()

        pos = pm.portfolio.get_position("000001")
        assert pos.available_quantity == 1000  # 次日可卖

    def test_update_prices(self, portfolio_manager: PortfolioManager) -> None:
        """测试价格更新"""
        pm = portfolio_manager

        buy_order = make_buy_order(quantity=1000, price=10.0)
        buy_order.filled_price = 10.0
        buy_order.filled_quantity = 1000
        buy_order.commission = 5.0
        buy_order.tax = 0.0
        buy_order.status = OrderStatus.FILLED
        pm.process_buy(buy_order)

        pm.update_prices({"000001": 12.0})

        pos = pm.portfolio.get_position("000001")
        assert pos.current_price == 12.0
        assert pos.market_value == 12_000.0

    def test_get_snapshot(self, portfolio_manager: PortfolioManager) -> None:
        """测试组合快照"""
        pm = portfolio_manager
        snapshot = pm.get_snapshot()

        assert snapshot["cash"] == 1_000_000.0
        assert snapshot["total_value"] == 1_000_000.0
        assert snapshot["market_value"] == 0.0
        assert len(snapshot["positions"]) == 0

    def test_pnl_calculation_loss(
        self, portfolio_manager: PortfolioManager
    ) -> None:
        """测试亏损PnL计算"""
        pm = portfolio_manager

        buy_order = make_buy_order(quantity=1000, price=10.0)
        buy_order.filled_price = 10.0
        buy_order.filled_quantity = 1000
        buy_order.commission = 5.0
        buy_order.tax = 0.0
        buy_order.status = OrderStatus.FILLED
        pm.process_buy(buy_order)
        pm.new_trading_day()

        # 卖出亏损（跌到9元）
        sell_order = make_sell_order(quantity=1000, price=9.0)
        sell_order.filled_price = 9.0
        sell_order.filled_quantity = 1000
        sell_order.commission = 5.0
        sell_order.tax = 9.0  # 印花税
        sell_order.status = OrderStatus.FILLED
        trade = pm.process_sell(sell_order)

        # PnL = (9.0 - 10.0) * 1000 - 5.0 - 9.0 = -1000 - 14 = -1014
        expected_pnl = (9.0 - 10.0) * 1000 - 5.0 - 9.0
        assert trade.pnl < 0
        assert abs(trade.pnl - expected_pnl) < 0.01


# ============================================================
# PerformanceMetrics Tests
# ============================================================


class TestPerformanceMetrics:
    """绩效指标测试"""

    def _make_equity_curve(
        self,
        start_value: float = 1_000_000.0,
        daily_returns: list[float] | None = None,
        n_days: int = 244,
    ) -> pd.Series:
        """构建净值曲线"""
        dates = pd.bdate_range(start="2024-01-01", periods=n_days)
        if daily_returns is not None:
            values = [start_value]
            for r in daily_returns:
                values.append(values[-1] * (1 + r))
            return pd.Series(values[: len(dates)], index=dates[: len(values)])
        else:
            # 每天涨0.05%
            values = [start_value * (1.0005**i) for i in range(n_days)]
            return pd.Series(values, index=dates)

    def test_total_return(self) -> None:
        """测试总收益率计算"""
        equity = pd.Series(
            [1_000_000.0, 1_100_000.0, 1_200_000.0],
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        config = BacktestConfig(initial_capital=1_000_000.0)
        result = PerformanceMetrics.calculate(equity, [], config)

        # 总收益率 = (1.2M - 1M) / 1M = 20%
        assert abs(result.total_return - 0.20) < 0.001

    def test_max_drawdown(self) -> None:
        """测试最大回撤计算"""
        # 先涨后跌: 100 -> 120 -> 90 -> 100
        equity = pd.Series(
            [100.0, 110.0, 120.0, 100.0, 90.0, 95.0, 100.0],
            index=pd.bdate_range("2024-01-01", periods=7),
        )
        config = BacktestConfig(initial_capital=100.0)
        result = PerformanceMetrics.calculate(equity, [], config)

        # 最大回撤 = (120 - 90) / 120 = 25%
        assert abs(result.max_drawdown - 0.25) < 0.001

    def test_max_drawdown_duration(self) -> None:
        """测试最大回撤持续天数"""
        # 100 -> 120 -> 100 -> 90 -> 95 -> 110 -> 125
        equity = pd.Series(
            [100.0, 120.0, 100.0, 90.0, 95.0, 110.0, 125.0],
            index=pd.bdate_range("2024-01-01", periods=7),
        )
        config = BacktestConfig(initial_capital=100.0)
        result = PerformanceMetrics.calculate(equity, [], config)

        # 从120跌到90再恢复到125，回撤持续4天（index 2~5在回撤中）
        # index 2: 100 < 120 回撤中
        # index 3: 90 < 120 回撤中
        # index 4: 95 < 120 回撤中
        # index 5: 110 < 120 回撤中
        # index 6: 125 > 120 恢复
        assert result.max_drawdown_duration == 4

    def test_sharpe_ratio(self) -> None:
        """测试夏普比率计算"""
        # 创建244天、每天涨0.05%的数据
        n_days = 244
        daily_ret = 0.0005
        start_val = 1_000_000.0
        values = [start_val * ((1 + daily_ret) ** i) for i in range(n_days)]
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        equity = pd.Series(values, index=dates)

        config = BacktestConfig(
            initial_capital=start_val,
            risk_free_rate=0.025,
        )
        result = PerformanceMetrics.calculate(equity, [], config)

        # 年化收益率 ≈ (1.0005)^244 - 1 ≈ 12.97%
        # 波动率很小（几乎为0因为是恒定日收益）
        # 夏普比率应很高
        assert result.annualized_return > 0.10
        assert result.sharpe_ratio > 5.0  # 恒定收益率的夏普非常高

    def test_sharpe_ratio_negative(self) -> None:
        """测试夏普比率 - 收益低于无风险利率"""
        # 每天跌0.01%
        n_days = 100
        daily_ret = -0.0001
        start_val = 1_000_000.0
        values = [start_val * ((1 + daily_ret) ** i) for i in range(n_days)]
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        equity = pd.Series(values, index=dates)

        config = BacktestConfig(
            initial_capital=start_val,
            risk_free_rate=0.025,
        )
        result = PerformanceMetrics.calculate(equity, [], config)

        assert result.sharpe_ratio < 0

    def test_trade_stats_win_rate(self) -> None:
        """测试交易统计 - 胜率"""
        trades = [
            TradeRecord("t1", date(2024, 1, 1), "000001", "sell", 100, 11.0, 5.0, 1.0, pnl=100.0),
            TradeRecord("t2", date(2024, 1, 2), "000002", "sell", 100, 9.0, 5.0, 1.0, pnl=-200.0),
            TradeRecord("t3", date(2024, 1, 3), "000003", "sell", 100, 12.0, 5.0, 1.0, pnl=300.0),
        ]
        equity = pd.Series(
            [1_000_000.0, 1_000_100.0, 999_900.0, 1_000_200.0],
            index=pd.bdate_range("2024-01-01", periods=4),
        )
        config = BacktestConfig(initial_capital=1_000_000.0)
        result = PerformanceMetrics.calculate(equity, trades, config)

        # 3笔交易，2盈1亏
        assert result.total_trades == 3
        assert abs(result.win_rate - 2 / 3) < 0.001

    def test_trade_stats_profit_loss_ratio(self) -> None:
        """测试交易统计 - 盈亏比"""
        trades = [
            TradeRecord("t1", date(2024, 1, 1), "000001", "sell", 100, 11.0, 5.0, 1.0, pnl=200.0),
            TradeRecord("t2", date(2024, 1, 2), "000002", "sell", 100, 9.0, 5.0, 1.0, pnl=-100.0),
        ]
        equity = pd.Series(
            [1_000_000.0, 1_000_200.0, 1_000_100.0],
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        config = BacktestConfig(initial_capital=1_000_000.0)
        result = PerformanceMetrics.calculate(equity, trades, config)

        # 盈亏比 = 200 / 100 = 2.0
        assert abs(result.profit_loss_ratio - 2.0) < 0.001
        assert abs(result.avg_win - 200.0) < 0.001
        assert abs(result.avg_loss - (-100.0)) < 0.001

    def test_sortino_ratio(self) -> None:
        """测试Sortino比率"""
        # 混合正负收益
        n_days = 244
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, n_days - 1)
        start_val = 1_000_000.0
        values = [start_val]
        for r in daily_returns:
            values.append(values[-1] * (1 + r))
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        equity = pd.Series(values, index=dates)

        config = BacktestConfig(initial_capital=start_val, risk_free_rate=0.025)
        result = PerformanceMetrics.calculate(equity, [], config)

        # Sortino应该被计算出来（可能正可能负）
        assert result.sortino_ratio != 0.0

    def test_calmar_ratio(self) -> None:
        """测试Calmar比率"""
        equity = pd.Series(
            [100.0, 110.0, 120.0, 100.0, 90.0, 130.0],
            index=pd.bdate_range("2024-01-01", periods=6),
        )
        config = BacktestConfig(initial_capital=100.0, risk_free_rate=0.025)
        result = PerformanceMetrics.calculate(equity, [], config)

        # max_drawdown = (120-90)/120 = 25%
        assert abs(result.max_drawdown - 0.25) < 0.001
        # calmar = annualized_return / 0.25
        expected_calmar = result.annualized_return / 0.25
        assert abs(result.calmar_ratio - expected_calmar) < 0.001

    def test_empty_equity_curve(self) -> None:
        """测试空净值曲线"""
        result = PerformanceMetrics.calculate(
            pd.Series(dtype=float),
            [],
            BacktestConfig(initial_capital=1_000_000.0),
        )
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0

    def test_summary_output(self) -> None:
        """测试绩效摘要输出"""
        equity = pd.Series(
            [1_000_000.0, 1_050_000.0, 1_100_000.0],
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        config = BacktestConfig(initial_capital=1_000_000.0)
        result = PerformanceMetrics.calculate(equity, [], config)
        summary = result.summary()

        assert "回测绩效报告" in summary
        assert "总收益率" in summary
        assert "夏普比率" in summary


# ============================================================
# BacktestEngine Tests
# ============================================================


class SimpleStrategy:
    """简单策略用于测试：第5天买入，第15天卖出"""

    def __init__(self) -> None:
        self.day_count = 0

    def on_bar(
        self,
        current_date: date,
        row: pd.Series,
        portfolio_snapshot: dict,
    ) -> Signal | None:
        self.day_count += 1

        if self.day_count == 5:
            return Signal(
                date=current_date,
                symbol="000001",
                signal_type=SignalType.BUY,
                price=row["close"],
            )
        elif self.day_count == 15:
            return Signal(
                date=current_date,
                symbol="000001",
                signal_type=SignalType.SELL,
                price=row["close"],
            )
        return None


class BuyAndHoldStrategy:
    """买入持有策略：第一天买入，最后一天卖出"""

    def __init__(self) -> None:
        self.bought = False

    def on_bar(
        self,
        current_date: date,
        row: pd.Series,
        portfolio_snapshot: dict,
    ) -> Signal | None:
        if not self.bought and len(portfolio_snapshot["positions"]) == 0:
            self.bought = True
            return Signal(
                date=current_date,
                symbol="000001",
                signal_type=SignalType.BUY,
                price=row["close"],
            )
        return None


def make_synthetic_data(
    n_days: int = 60,
    start_price: float = 10.0,
    daily_return: float = 0.002,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """生成合成行情数据"""
    dates = pd.bdate_range(start=start_date, periods=n_days)
    prices = [start_price]
    for i in range(1, n_days):
        prices.append(prices[-1] * (1 + daily_return))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": [p * 0.998 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000] * n_days,
        }
    )
    return df


class TestBacktestEngine:
    """回测引擎集成测试"""

    def test_engine_run_basic(self, app_config: AppConfig) -> None:
        """测试基本回测流程"""
        engine = BacktestEngine(app_config)
        strategy = SimpleStrategy()
        data = make_synthetic_data(n_days=30, start_date="2024-01-01")

        result = engine.run(strategy, data, "000001")

        assert isinstance(result, PerformanceResult)
        assert not result.equity_curve.empty
        assert len(result.equity_curve) > 0

    def test_engine_buy_and_sell(self, app_config: AppConfig) -> None:
        """测试买卖完整流程"""
        engine = BacktestEngine(app_config)
        strategy = SimpleStrategy()
        data = make_synthetic_data(
            n_days=30, start_price=10.0, daily_return=0.005, start_date="2024-01-01"
        )

        result = engine.run(strategy, data, "000001")

        # 应有至少1笔交易
        assert len(engine.trades) >= 1
        # 交易应有pnl
        assert engine.trades[0].pnl != 0.0

    def test_engine_equity_curve_recorded(self, app_config: AppConfig) -> None:
        """测试净值曲线记录"""
        engine = BacktestEngine(app_config)
        strategy = BuyAndHoldStrategy()
        data = make_synthetic_data(n_days=20, start_date="2024-01-01")

        result = engine.run(strategy, data, "000001")

        assert len(result.equity_curve) == 20

    def test_engine_no_trade_strategy(self, app_config: AppConfig) -> None:
        """测试无交易策略"""

        class NoTradeStrategy:
            def on_bar(self, d, row, snap):
                return None

        engine = BacktestEngine(app_config)
        data = make_synthetic_data(n_days=10, start_date="2024-01-01")
        result = engine.run(NoTradeStrategy(), data, "000001")

        # 无交易，总收益为0
        assert abs(result.total_return) < 0.0001
        assert result.total_trades == 0

    def test_engine_with_risk_manager(self, app_config: AppConfig) -> None:
        """测试带风险管理的回测"""
        engine = BacktestEngine(app_config)
        strategy = SimpleStrategy()
        data = make_synthetic_data(
            n_days=30, start_price=10.0, daily_return=0.005, start_date="2024-01-01"
        )

        # 触发风险管理器初始化
        engine._init_risk_manager()
        assert engine.risk_manager is not None

        result = engine.run(strategy, data, "000001")
        assert isinstance(result, PerformanceResult)

    def test_engine_reset_between_runs(self, app_config: AppConfig) -> None:
        """测试两次回测之间状态重置"""
        engine = BacktestEngine(app_config)
        strategy1 = SimpleStrategy()
        strategy2 = SimpleStrategy()
        data = make_synthetic_data(n_days=30, start_date="2024-01-01")

        result1 = engine.run(strategy1, data, "000001")
        result2 = engine.run(strategy2, data, "000001")

        # 两次结果应相同（策略行为一致）
        assert abs(result1.total_return - result2.total_return) < 0.0001
