"""风险管理模块测试

测试覆盖：
- PositionSizer: 固定比例仓位、手数取整、最大持仓检查、Kelly公式、ATR法
- StopLossManager: 止损触发、止盈触发
- RiskManager: 信号验证、持仓限制、退出检查
"""

from __future__ import annotations

from datetime import date

import pytest

from quant_trading.core.config import RiskConfig
from quant_trading.core.enums import SignalType
from quant_trading.core.models import Portfolio, Position, Signal
from quant_trading.risk.manager import RiskManager
from quant_trading.risk.position_sizer import PositionSizer
from quant_trading.risk.stop_loss import StopLossManager


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_pct=0.3,
        max_total_position_pct=0.8,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        position_sizing_method="fixed_ratio",
        fixed_ratio=0.1,
    )


@pytest.fixture
def position_sizer(risk_config: RiskConfig) -> PositionSizer:
    return PositionSizer(risk_config)


@pytest.fixture
def stop_loss_manager(risk_config: RiskConfig) -> StopLossManager:
    return StopLossManager(risk_config)


@pytest.fixture
def risk_manager(risk_config: RiskConfig) -> RiskManager:
    return RiskManager(risk_config)


def make_signal(
    symbol: str = "000001",
    signal_type: str = SignalType.BUY,
    price: float = 10.0,
    strength: float = 1.0,
) -> Signal:
    return Signal(
        date=date(2024, 3, 1),
        symbol=symbol,
        signal_type=signal_type,
        price=price,
        strength=strength,
    )


def make_portfolio(
    cash: float = 1_000_000.0,
    positions: dict | None = None,
) -> Portfolio:
    portfolio = Portfolio(cash=cash, initial_capital=1_000_000.0)
    if positions:
        for symbol, pos_data in positions.items():
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=pos_data.get("quantity", 0),
                available_quantity=pos_data.get("available_quantity", 0),
                avg_cost=pos_data.get("avg_cost", 0.0),
                current_price=pos_data.get("current_price", 0.0),
                buy_date=date(2024, 3, 1),
            )
    return portfolio


def make_position(
    symbol: str = "000001",
    quantity: int = 1000,
    available_quantity: int = 1000,
    avg_cost: float = 10.0,
    current_price: float = 10.0,
) -> Position:
    return Position(
        symbol=symbol,
        quantity=quantity,
        available_quantity=available_quantity,
        avg_cost=avg_cost,
        current_price=current_price,
        buy_date=date(2024, 3, 1),
    )


# ============================================================
# PositionSizer Tests
# ============================================================


class TestPositionSizer:
    """仓位管理器测试"""

    def test_fixed_ratio_sizing(self, position_sizer: PositionSizer) -> None:
        """测试固定比例法"""
        signal = make_signal(price=10.0)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty = position_sizer.calculate_quantity(signal, portfolio)

        # 总资产 1M * 10% = 100,000 / 10.0 = 10,000股
        # 向下取整到100 -> 10,000
        assert qty == 10_000

    def test_fixed_ratio_sizing_odd_quantity(
        self, position_sizer: PositionSizer
    ) -> None:
        """测试固定比例法 - 非整数手取整"""
        signal = make_signal(price=13.7)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty = position_sizer.calculate_quantity(signal, portfolio)

        # 1M * 10% = 100,000 / 13.7 = 7299.27 -> 7200
        assert qty % 100 == 0
        expected = (int(100_000 / 13.7) // 100) * 100
        assert qty == expected

    def test_lot_size_rounding(self, position_sizer: PositionSizer) -> None:
        """测试手数取整 - 始终是100的整数倍"""
        signal = make_signal(price=33.33)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty = position_sizer.calculate_quantity(signal, portfolio)

        assert qty % 100 == 0
        assert qty > 0

    def test_max_position_limit(self, position_sizer: PositionSizer) -> None:
        """测试最大持仓比例限制"""
        signal = make_signal(symbol="000001", price=10.0)

        # 已有25%的持仓（接近30%上限）
        portfolio = make_portfolio(
            cash=750_000.0,
            positions={
                "000001": {
                    "quantity": 25_000,
                    "available_quantity": 25_000,
                    "avg_cost": 10.0,
                    "current_price": 10.0,
                }
            },
        )

        qty = position_sizer.calculate_quantity(signal, portfolio)

        # 已有250k市值 / 1M总值 = 25%
        # 上限30% = 300k -> 最多再买50k / 10 = 5000股
        total_value = portfolio.total_value
        max_add = total_value * 0.3 - 250_000
        max_qty = (int(max_add / 10.0) // 100) * 100
        assert qty <= max_qty

    def test_max_position_already_at_limit(
        self, position_sizer: PositionSizer
    ) -> None:
        """测试最大持仓比例 - 已达上限"""
        signal = make_signal(symbol="000001", price=10.0)

        # 已有30%持仓
        portfolio = make_portfolio(
            cash=700_000.0,
            positions={
                "000001": {
                    "quantity": 30_000,
                    "available_quantity": 30_000,
                    "avg_cost": 10.0,
                    "current_price": 10.0,
                }
            },
        )

        qty = position_sizer.calculate_quantity(signal, portfolio)
        assert qty == 0

    def test_cash_limit(self, position_sizer: PositionSizer) -> None:
        """测试现金不足限制"""
        signal = make_signal(price=10.0)
        # 只有5000元现金
        portfolio = make_portfolio(cash=5_000.0)

        qty = position_sizer.calculate_quantity(signal, portfolio)

        # 10% * 5000 = 500 / 10 = 50 -> 不足100股 = 0
        # 但cash limit: 5000 / 10 = 500 -> 500
        assert qty <= 500
        assert qty % 100 == 0

    def test_zero_price(self, position_sizer: PositionSizer) -> None:
        """测试价格为零"""
        signal = make_signal(price=0.0)
        portfolio = make_portfolio()
        qty = position_sizer.calculate_quantity(signal, portfolio)
        assert qty == 0

    def test_signal_strength_adjustment(
        self, position_sizer: PositionSizer
    ) -> None:
        """测试信号强度调整"""
        signal_full = make_signal(price=10.0, strength=1.0)
        signal_half = make_signal(price=10.0, strength=0.5)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty_full = position_sizer.calculate_quantity(signal_full, portfolio)
        qty_half = position_sizer.calculate_quantity(signal_half, portfolio)

        # 半强度信号的数量应该约为全强度的一半
        assert qty_half <= qty_full
        assert qty_half > 0

    def test_kelly_sizing_positive(self, position_sizer: PositionSizer) -> None:
        """测试Kelly公式 - 正值"""
        position_sizer.update_kelly_params(win_rate=0.6, profit_loss_ratio=2.0)

        signal = make_signal(price=10.0)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty = position_sizer.calculate_quantity(
            signal, portfolio, method="kelly"
        )

        # kelly% = 0.6 - 0.4/2.0 = 0.6 - 0.2 = 0.4
        # half_kelly = 0.2
        # 但被 fixed_ratio * 2 = 0.2 限制
        # 1M * 0.2 / 10 = 20,000
        assert qty > 0
        assert qty % 100 == 0

    def test_kelly_sizing_negative(self, position_sizer: PositionSizer) -> None:
        """测试Kelly公式 - 负值（不应交易）"""
        position_sizer.update_kelly_params(win_rate=0.3, profit_loss_ratio=0.5)

        signal = make_signal(price=10.0)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty = position_sizer.calculate_quantity(
            signal, portfolio, method="kelly"
        )

        # kelly% = 0.3 - 0.7/0.5 = 0.3 - 1.4 = -1.1 < 0
        assert qty == 0

    def test_atr_sizing(self, position_sizer: PositionSizer) -> None:
        """测试ATR自适应法"""
        position_sizer.update_atr(0.5)  # ATR = 0.5
        position_sizer.set_risk_pct(0.02)

        signal = make_signal(price=10.0)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty = position_sizer.calculate_quantity(
            signal, portfolio, method="atr_based"
        )

        # risk_per_share = 0.5 * 2 = 1.0
        # risk_capital = 1M * 0.02 = 20,000
        # quantity = 20,000 / 1.0 = 20,000
        # 但受max_position_pct 30%限制: 1M * 30% / 10 = 30,000 -> 不限制
        assert qty > 0
        assert qty % 100 == 0

    def test_atr_sizing_fallback(self, position_sizer: PositionSizer) -> None:
        """测试ATR未设置时回退到固定比例"""
        # ATR为0，应回退
        signal = make_signal(price=10.0)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty_atr = position_sizer.calculate_quantity(
            signal, portfolio, method="atr_based"
        )
        qty_fixed = position_sizer.calculate_quantity(
            signal, portfolio, method="fixed_ratio"
        )

        assert qty_atr == qty_fixed


# ============================================================
# StopLossManager Tests
# ============================================================


class TestStopLossManager:
    """止损止盈管理器测试"""

    def test_stop_loss_triggered(self, stop_loss_manager: StopLossManager) -> None:
        """测试止损触发"""
        # 成本10, 止损5%, 止损价=9.5, 当前价9.4 -> 触发
        position = make_position(avg_cost=10.0, current_price=9.4)
        assert stop_loss_manager.check_stop_loss(position) is True

    def test_stop_loss_not_triggered(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试止损未触发"""
        # 成本10, 止损5%, 止损价=9.5, 当前价9.6 -> 未触发
        position = make_position(avg_cost=10.0, current_price=9.6)
        assert stop_loss_manager.check_stop_loss(position) is False

    def test_stop_loss_exact_boundary(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试止损边界值"""
        # 成本10, 止损5%, 止损价=9.5, 当前价=9.5 -> 触发（<=）
        position = make_position(avg_cost=10.0, current_price=9.5)
        assert stop_loss_manager.check_stop_loss(position) is True

    def test_take_profit_triggered(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试止盈触发"""
        # 成本10, 止盈15%, 止盈价=11.5, 当前价11.6 -> 触发
        position = make_position(avg_cost=10.0, current_price=11.6)
        assert stop_loss_manager.check_take_profit(position) is True

    def test_take_profit_not_triggered(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试止盈未触发"""
        # 成本10, 止盈15%, 止盈价=11.5, 当前价11.4 -> 未触发
        position = make_position(avg_cost=10.0, current_price=11.4)
        assert stop_loss_manager.check_take_profit(position) is False

    def test_take_profit_exact_boundary(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试止盈边界值"""
        # 成本10, 止盈15%, 止盈价=11.5, 当前价=11.5 -> 触发（>=）
        position = make_position(avg_cost=10.0, current_price=11.5)
        assert stop_loss_manager.check_take_profit(position) is True

    def test_should_exit_stop_loss(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试should_exit - 止损"""
        position = make_position(avg_cost=10.0, current_price=9.0)
        should_exit, reason = stop_loss_manager.should_exit(position)

        assert should_exit is True
        assert "止损" in reason

    def test_should_exit_take_profit(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试should_exit - 止盈"""
        position = make_position(avg_cost=10.0, current_price=12.0)
        should_exit, reason = stop_loss_manager.should_exit(position)

        assert should_exit is True
        assert "止盈" in reason

    def test_should_exit_neither(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试should_exit - 未触发"""
        position = make_position(avg_cost=10.0, current_price=10.5)
        should_exit, reason = stop_loss_manager.should_exit(position)

        assert should_exit is False
        assert reason == ""

    def test_stop_loss_zero_cost(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试零成本持仓不触发止损"""
        position = make_position(avg_cost=0.0, current_price=5.0)
        assert stop_loss_manager.check_stop_loss(position) is False

    def test_get_stop_price(self, stop_loss_manager: StopLossManager) -> None:
        """测试获取止损价"""
        position = make_position(avg_cost=10.0)
        stop_price = stop_loss_manager.get_stop_price(position)
        # 10 * (1 - 0.05) = 9.5
        assert abs(stop_price - 9.5) < 0.001

    def test_get_take_profit_price(
        self, stop_loss_manager: StopLossManager
    ) -> None:
        """测试获取止盈价"""
        position = make_position(avg_cost=10.0)
        tp_price = stop_loss_manager.get_take_profit_price(position)
        # 10 * (1 + 0.15) = 11.5
        assert abs(tp_price - 11.5) < 0.001


# ============================================================
# RiskManager Tests
# ============================================================


class TestRiskManager:
    """风险管理器测试"""

    def test_validate_buy_signal_pass(self, risk_manager: RiskManager) -> None:
        """测试买入信号验证 - 通过"""
        signal = make_signal(signal_type=SignalType.BUY)
        portfolio = make_portfolio(cash=1_000_000.0)

        valid, reason = risk_manager.validate_signal(signal, portfolio)
        assert valid is True

    def test_validate_sell_signal_always_pass(
        self, risk_manager: RiskManager
    ) -> None:
        """测试卖出信号始终通过"""
        signal = make_signal(signal_type=SignalType.SELL)
        portfolio = make_portfolio(cash=0.0)  # 即使没钱也通过

        valid, reason = risk_manager.validate_signal(signal, portfolio)
        assert valid is True

    def test_validate_single_position_limit(
        self, risk_manager: RiskManager
    ) -> None:
        """测试单只股票持仓比例限制"""
        signal = make_signal(symbol="000001", signal_type=SignalType.BUY, price=10.0)

        # 已有30%持仓（达到上限）
        portfolio = make_portfolio(
            cash=700_000.0,
            positions={
                "000001": {
                    "quantity": 30_000,
                    "available_quantity": 30_000,
                    "avg_cost": 10.0,
                    "current_price": 10.0,
                }
            },
        )

        valid, reason = risk_manager.validate_signal(signal, portfolio)
        assert valid is False
        assert "持仓比例" in reason

    def test_validate_total_position_limit(
        self, risk_manager: RiskManager
    ) -> None:
        """测试总持仓比例限制"""
        signal = make_signal(symbol="000003", signal_type=SignalType.BUY, price=10.0)

        # 总持仓80%（达到上限）
        portfolio = make_portfolio(
            cash=200_000.0,
            positions={
                "000001": {
                    "quantity": 40_000,
                    "available_quantity": 40_000,
                    "avg_cost": 10.0,
                    "current_price": 10.0,
                },
                "000002": {
                    "quantity": 40_000,
                    "available_quantity": 40_000,
                    "avg_cost": 10.0,
                    "current_price": 10.0,
                },
            },
        )

        valid, reason = risk_manager.validate_signal(signal, portfolio)
        assert valid is False
        assert "总持仓比例" in reason

    def test_validate_below_limits(self, risk_manager: RiskManager) -> None:
        """测试持仓低于限制"""
        signal = make_signal(symbol="000002", signal_type=SignalType.BUY, price=10.0)

        portfolio = make_portfolio(
            cash=800_000.0,
            positions={
                "000001": {
                    "quantity": 10_000,
                    "available_quantity": 10_000,
                    "avg_cost": 10.0,
                    "current_price": 10.0,
                }
            },
        )

        valid, reason = risk_manager.validate_signal(signal, portfolio)
        assert valid is True

    def test_calculate_order_quantity(self, risk_manager: RiskManager) -> None:
        """测试下单数量计算"""
        signal = make_signal(price=10.0)
        portfolio = make_portfolio(cash=1_000_000.0)

        qty = risk_manager.calculate_order_quantity(signal, portfolio)

        assert qty > 0
        assert qty % 100 == 0

    def test_check_exits_stop_loss(self, risk_manager: RiskManager) -> None:
        """测试退出检查 - 止损"""
        portfolio = make_portfolio(
            cash=900_000.0,
            positions={
                "000001": {
                    "quantity": 1000,
                    "available_quantity": 1000,
                    "avg_cost": 10.0,
                    "current_price": 9.0,  # 跌10%，超过5%止损
                }
            },
        )

        exit_signals = risk_manager.check_exits(portfolio)

        assert len(exit_signals) == 1
        assert exit_signals[0].symbol == "000001"
        assert exit_signals[0].signal_type == SignalType.SELL

    def test_check_exits_take_profit(self, risk_manager: RiskManager) -> None:
        """测试退出检查 - 止盈"""
        portfolio = make_portfolio(
            cash=900_000.0,
            positions={
                "000001": {
                    "quantity": 1000,
                    "available_quantity": 1000,
                    "avg_cost": 10.0,
                    "current_price": 12.0,  # 涨20%，超过15%止盈
                }
            },
        )

        exit_signals = risk_manager.check_exits(portfolio)

        assert len(exit_signals) == 1
        assert exit_signals[0].symbol == "000001"

    def test_check_exits_t_plus_1_skipped(
        self, risk_manager: RiskManager
    ) -> None:
        """测试退出检查 - T+1当日买入不触发"""
        portfolio = make_portfolio(
            cash=900_000.0,
            positions={
                "000001": {
                    "quantity": 1000,
                    "available_quantity": 0,  # 当日买入，不可卖
                    "avg_cost": 10.0,
                    "current_price": 9.0,  # 已触发止损
                }
            },
        )

        exit_signals = risk_manager.check_exits(portfolio)

        # 虽然触发止损，但因T+1不可卖，不生成退出信号
        assert len(exit_signals) == 0

    def test_check_exits_no_trigger(self, risk_manager: RiskManager) -> None:
        """测试退出检查 - 无触发"""
        portfolio = make_portfolio(
            cash=900_000.0,
            positions={
                "000001": {
                    "quantity": 1000,
                    "available_quantity": 1000,
                    "avg_cost": 10.0,
                    "current_price": 10.5,  # 微涨，不触发
                }
            },
        )

        exit_signals = risk_manager.check_exits(portfolio)
        assert len(exit_signals) == 0

    def test_check_exits_multiple_positions(
        self, risk_manager: RiskManager
    ) -> None:
        """测试退出检查 - 多只股票"""
        portfolio = make_portfolio(
            cash=800_000.0,
            positions={
                "000001": {
                    "quantity": 1000,
                    "available_quantity": 1000,
                    "avg_cost": 10.0,
                    "current_price": 9.0,  # 止损
                },
                "000002": {
                    "quantity": 1000,
                    "available_quantity": 1000,
                    "avg_cost": 10.0,
                    "current_price": 12.0,  # 止盈
                },
                "000003": {
                    "quantity": 1000,
                    "available_quantity": 1000,
                    "avg_cost": 10.0,
                    "current_price": 10.5,  # 不触发
                },
            },
        )

        exit_signals = risk_manager.check_exits(portfolio)

        assert len(exit_signals) == 2
        symbols = {s.symbol for s in exit_signals}
        assert "000001" in symbols
        assert "000002" in symbols
        assert "000003" not in symbols

    def test_risk_report(self, risk_manager: RiskManager) -> None:
        """测试风险报告"""
        portfolio = make_portfolio(
            cash=900_000.0,
            positions={
                "000001": {
                    "quantity": 1000,
                    "available_quantity": 1000,
                    "avg_cost": 10.0,
                    "current_price": 10.5,
                }
            },
        )

        report = risk_manager.get_risk_report(portfolio)

        assert "total_value" in report
        assert "total_position_pct" in report
        assert "positions_risk" in report
        assert "000001" in report["positions_risk"]

    def test_update_kelly_params(self, risk_manager: RiskManager) -> None:
        """测试Kelly参数更新"""
        risk_manager.update_kelly_params(0.65, 2.5)

        sizer = risk_manager.position_sizer
        assert sizer._win_rate == 0.65
        assert sizer._profit_loss_ratio == 2.5

    def test_update_atr(self, risk_manager: RiskManager) -> None:
        """测试ATR更新"""
        risk_manager.update_atr(0.75)
        assert risk_manager.position_sizer._last_atr == 0.75
