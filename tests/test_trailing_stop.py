"""追踪止损与涨跌停测试"""

from __future__ import annotations

import pytest

from quant_trading.risk.stop_loss import (
    TrailingStopManager,
    TrailingStopMode,
)
from quant_trading.backtest.broker import PriceLimitChecker


# =============================================================================
# TrailingStopManager 测试
# =============================================================================

class TestTrailingStopPercent:
    """固定百分比追踪止损"""

    def setup_method(self):
        self.ts = TrailingStopManager(
            mode=TrailingStopMode.PERCENT, trail_pct=0.05
        )

    def test_initial_stop(self):
        """首次更新应设置止损价"""
        stop = self.ts.update("000001", 10.0, avg_cost=10.0)
        assert stop == pytest.approx(10.0 * 0.95, rel=1e-6)

    def test_price_rises_stop_rises(self):
        """价格上涨时止损价应跟随上移"""
        self.ts.update("000001", 10.0, avg_cost=10.0)
        stop = self.ts.update("000001", 12.0, avg_cost=10.0)
        assert stop == pytest.approx(12.0 * 0.95, rel=1e-6)

    def test_price_drops_stop_stays(self):
        """价格下跌时止损价不应下移"""
        self.ts.update("000001", 12.0, avg_cost=10.0)
        stop = self.ts.update("000001", 11.0, avg_cost=10.0)
        # 止损价仍然基于最高价12.0
        assert stop == pytest.approx(12.0 * 0.95, rel=1e-6)

    def test_trigger(self):
        """价格跌破止损价应触发"""
        self.ts.update("000001", 12.0, avg_cost=10.0)
        # 止损价 = 12 * 0.95 = 11.4
        triggered, stop = self.ts.check("000001", 11.3, avg_cost=10.0)
        assert triggered is True
        assert stop == pytest.approx(12.0 * 0.95, rel=1e-6)

    def test_no_trigger(self):
        """价格高于止损价不应触发"""
        self.ts.update("000001", 12.0, avg_cost=10.0)
        triggered, stop = self.ts.check("000001", 11.5, avg_cost=10.0)
        assert triggered is False

    def test_reset(self):
        """重置应清除所有追踪状态"""
        self.ts.update("000001", 12.0, avg_cost=10.0)
        self.ts.reset("000001")
        assert self.ts.get_stop_price("000001") is None
        assert self.ts.get_high_watermark("000001") is None

    def test_reset_all(self):
        """重置全部"""
        self.ts.update("000001", 12.0, avg_cost=10.0)
        self.ts.update("600519", 200.0, avg_cost=180.0)
        self.ts.reset()
        assert self.ts.get_stop_price("000001") is None
        assert self.ts.get_stop_price("600519") is None

    def test_multiple_symbols(self):
        """多只股票独立追踪"""
        self.ts.update("000001", 10.0, avg_cost=10.0)
        self.ts.update("600519", 200.0, avg_cost=180.0)
        assert self.ts.get_high_watermark("000001") == 10.0
        assert self.ts.get_high_watermark("600519") == 200.0


class TestTrailingStopATR:
    """ATR倍数追踪止损"""

    def test_atr_mode(self):
        ts = TrailingStopManager(
            mode=TrailingStopMode.ATR, atr_multiplier=2.0
        )
        stop = ts.update("000001", 10.0, avg_cost=10.0, atr=0.3)
        # stop = 10.0 - 2.0 * 0.3 = 9.4
        assert stop == pytest.approx(9.4, rel=1e-6)

    def test_atr_none_fallback(self):
        """ATR为None时回退到百分比模式"""
        ts = TrailingStopManager(
            mode=TrailingStopMode.ATR, trail_pct=0.05
        )
        stop = ts.update("000001", 10.0, avg_cost=10.0, atr=None)
        assert stop == pytest.approx(10.0 * 0.95, rel=1e-6)

    def test_atr_stop_only_rises(self):
        ts = TrailingStopManager(
            mode=TrailingStopMode.ATR, atr_multiplier=2.0
        )
        ts.update("000001", 10.0, avg_cost=10.0, atr=0.3)
        # 价格上升
        stop = ts.update("000001", 12.0, avg_cost=10.0, atr=0.3)
        # stop = 12.0 - 0.6 = 11.4
        assert stop == pytest.approx(11.4, rel=1e-6)
        # 价格回落，止损不回退
        stop = ts.update("000001", 11.0, avg_cost=10.0, atr=0.3)
        assert stop == pytest.approx(11.4, rel=1e-6)


class TestTrailingStopStep:
    """阶梯式追踪止损"""

    def test_step_low_profit(self):
        """低盈利使用宽松止损"""
        ts = TrailingStopManager(
            mode=TrailingStopMode.STEP,
            steps=[
                (0.00, 0.05),
                (0.10, 0.03),
                (0.20, 0.02),
            ],
        )
        # 成本10, 当前10.5, 盈利5% -> 使用 trail=5%
        stop = ts.update("000001", 10.5, avg_cost=10.0)
        assert stop == pytest.approx(10.5 * 0.95, rel=1e-6)

    def test_step_high_profit(self):
        """高盈利使用紧凑止损"""
        ts = TrailingStopManager(
            mode=TrailingStopMode.STEP,
            steps=[
                (0.00, 0.05),
                (0.10, 0.03),
                (0.20, 0.02),
            ],
        )
        # 成本10, 当前12.5, 盈利25% -> 使用 trail=2%
        stop = ts.update("000001", 12.5, avg_cost=10.0)
        assert stop == pytest.approx(12.5 * 0.98, rel=1e-6)


# =============================================================================
# PriceLimitChecker 涨跌停测试
# =============================================================================

class TestPriceLimitChecker:
    """涨跌停限制校验"""

    def setup_method(self):
        self.checker = PriceLimitChecker(enabled=True)

    def test_board_type_normal(self):
        assert self.checker.get_board_type("000001") == "normal"
        assert self.checker.get_board_type("600519") == "normal"

    def test_board_type_star(self):
        assert self.checker.get_board_type("688001") == "star"

    def test_board_type_gem(self):
        assert self.checker.get_board_type("300750") == "gem"

    def test_board_type_bse(self):
        assert self.checker.get_board_type("830799") == "bse"

    def test_board_type_st(self):
        assert self.checker.get_board_type("000001", is_st=True) == "st"

    def test_limit_ratio(self):
        assert self.checker.get_limit_ratio("000001") == 0.10
        assert self.checker.get_limit_ratio("688001") == 0.20
        assert self.checker.get_limit_ratio("300750") == 0.20
        assert self.checker.get_limit_ratio("000001", is_st=True) == 0.05
        assert self.checker.get_limit_ratio("830799") == 0.30

    def test_upper_limit(self):
        """涨停检测"""
        at_upper, at_lower, upper_p, lower_p = self.checker.check(
            "000001", 11.0, prev_close=10.0
        )
        assert at_upper is True
        assert at_lower is False
        assert upper_p == pytest.approx(11.0, rel=1e-6)

    def test_lower_limit(self):
        """跌停检测"""
        at_upper, at_lower, upper_p, lower_p = self.checker.check(
            "000001", 9.0, prev_close=10.0
        )
        assert at_upper is False
        assert at_lower is True
        assert lower_p == pytest.approx(9.0, rel=1e-6)

    def test_no_limit(self):
        """正常价格"""
        at_upper, at_lower, _, _ = self.checker.check(
            "000001", 10.5, prev_close=10.0
        )
        assert at_upper is False
        assert at_lower is False

    def test_can_buy_at_limit(self):
        """涨停时不可买入"""
        can, reason = self.checker.can_buy("000001", 11.0, prev_close=10.0)
        assert can is False
        assert "涨停" in reason

    def test_can_sell_at_limit(self):
        """跌停时不可卖出"""
        can, reason = self.checker.can_sell("000001", 9.0, prev_close=10.0)
        assert can is False
        assert "跌停" in reason

    def test_can_buy_normal(self):
        """正常价格可买入"""
        can, _ = self.checker.can_buy("000001", 10.5, prev_close=10.0)
        assert can is True

    def test_can_sell_normal(self):
        """正常价格可卖出"""
        can, _ = self.checker.can_sell("000001", 10.5, prev_close=10.0)
        assert can is True

    def test_disabled(self):
        """禁用时总是允许"""
        checker = PriceLimitChecker(enabled=False)
        can_buy, _ = checker.can_buy("000001", 11.0, prev_close=10.0)
        can_sell, _ = checker.can_sell("000001", 9.0, prev_close=10.0)
        assert can_buy is True
        assert can_sell is True

    def test_star_board_limit(self):
        """科创板 ±20%"""
        # 12.0 == 10.0 * 1.20 = 涨停价，已涨停不可买
        can, reason = self.checker.can_buy("688001", 12.0, prev_close=10.0)
        assert can is False
        # 未涨停可买
        can, reason = self.checker.can_buy("688001", 11.9, prev_close=10.0)
        assert can is True

    def test_st_limit(self):
        """ST股 ±5%"""
        # 10.5 == 10.0 * 1.05 = 涨停价，已涨停不可买
        can, _ = self.checker.can_buy("000001", 10.5, prev_close=10.0, is_st=True)
        assert can is False
        # 未涨停可买
        can, _ = self.checker.can_buy("000001", 10.49, prev_close=10.0, is_st=True)
        assert can is True
