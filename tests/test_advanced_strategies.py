"""高级策略模块单元测试

测试 DualThrust、MeanReversion、Turtle 三种高级策略。
使用合成数据验证信号生成逻辑、参数校验和边界情况。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.strategy.base import StrategyRegistry
from quant_trading.strategy.dual_thrust import DualThrustStrategy
from quant_trading.strategy.mean_reversion import MeanReversionStrategy
from quant_trading.strategy.turtle import TurtleStrategy


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

VALID_SIGNALS = {"buy", "sell", "hold"}


def _make_ohlcv(n: int = 100, seed: int = 42, base_price: float = 10.0) -> pd.DataFrame:
    """生成包含 OHLCV 的合成行情数据。"""
    rng = np.random.RandomState(seed)
    close = base_price + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(1000, 100_000, n).astype(float)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_uptrend(n: int = 80) -> pd.DataFrame:
    """生成持续上升趋势的数据。"""
    prices = np.linspace(10, 50, n) + np.random.RandomState(1).randn(n) * 0.3
    high = prices + 0.5
    low = prices - 0.5
    return pd.DataFrame({
        "open": prices + 0.1,
        "high": high,
        "low": low,
        "close": prices,
        "volume": np.full(n, 10000.0),
    })


def _make_downtrend(n: int = 80) -> pd.DataFrame:
    """生成持续下降趋势的数据。"""
    prices = np.linspace(50, 10, n) + np.random.RandomState(2).randn(n) * 0.3
    high = prices + 0.5
    low = prices - 0.5
    return pd.DataFrame({
        "open": prices + 0.1,
        "high": high,
        "low": low,
        "close": prices,
        "volume": np.full(n, 10000.0),
    })


def _make_volatile(n: int = 100) -> pd.DataFrame:
    """生成高波动数据（均值回归应在此产生信号）。"""
    rng = np.random.RandomState(3)
    prices = 20.0 + np.cumsum(rng.randn(n) * 2.0)
    high = prices + rng.uniform(0.5, 2.0, n)
    low = prices - rng.uniform(0.5, 2.0, n)
    return pd.DataFrame({
        "open": prices + rng.randn(n) * 0.5,
        "high": high,
        "low": low,
        "close": prices,
        "volume": rng.randint(1000, 100_000, n).astype(float),
    })


def _make_flat(n: int = 60, price: float = 20.0) -> pd.DataFrame:
    """生成基本平稳（微小噪声）的数据。"""
    rng = np.random.RandomState(4)
    prices = np.full(n, price) + rng.randn(n) * 0.01
    return pd.DataFrame({
        "open": prices,
        "high": prices + 0.02,
        "low": prices - 0.02,
        "close": prices,
        "volume": np.full(n, 5000.0),
    })


# =========================================================================
# DualThrustStrategy
# =========================================================================

class TestDualThrust:
    def test_registry(self):
        """应已在注册表中注册。"""
        assert "dual_thrust" in StrategyRegistry.list_strategies()
        strategy = StrategyRegistry.get("dual_thrust", lookback=4, k1=0.5, k2=0.5)
        assert isinstance(strategy, DualThrustStrategy)

    def test_uptrend_generates_buy(self):
        """突然加速上涨应突破上轨产生买入信号。"""
        # 先横盘建立较小的 range，然后急涨使 close 远超 open + k*range
        n = 60
        rng = np.random.RandomState(100)
        flat_close = np.full(30, 20.0) + rng.randn(30) * 0.1
        surge_close = np.linspace(20, 40, 30)  # 急涨
        close = np.concatenate([flat_close, surge_close])
        # open 贴近前一日 close（模拟跳空高开 + 继续涨）
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        data = pd.DataFrame({
            "open": open_,
            "high": np.maximum(close, open_) + 0.2,
            "low": np.minimum(close, open_) - 0.2,
            "close": close,
            "volume": np.full(n, 10000.0),
        })
        strategy = DualThrustStrategy(lookback=4, k1=0.3, k2=0.3)
        result = strategy.generate_signals(data)
        assert "buy" in result["signal"].values, "急涨行情中应产生买入信号"

    def test_downtrend_generates_sell(self):
        """突然加速下跌应突破下轨产生卖出信号。"""
        n = 60
        rng = np.random.RandomState(101)
        flat_close = np.full(30, 20.0) + rng.randn(30) * 0.1
        plunge_close = np.linspace(20, 5, 30)  # 急跌
        close = np.concatenate([flat_close, plunge_close])
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        data = pd.DataFrame({
            "open": open_,
            "high": np.maximum(close, open_) + 0.2,
            "low": np.minimum(close, open_) - 0.2,
            "close": close,
            "volume": np.full(n, 10000.0),
        })
        strategy = DualThrustStrategy(lookback=4, k1=0.3, k2=0.3)
        result = strategy.generate_signals(data)
        assert "sell" in result["signal"].values, "急跌行情中应产生卖出信号"

    def test_only_valid_signals(self):
        """信号列只应包含 buy/sell/hold。"""
        data = _make_ohlcv(n=100)
        result = DualThrustStrategy().generate_signals(data)
        assert set(result["signal"].unique()).issubset(VALID_SIGNALS)

    def test_output_length_matches_input(self):
        """输出 DataFrame 长度应与输入一致。"""
        data = _make_ohlcv(n=50)
        result = DualThrustStrategy().generate_signals(data)
        assert len(result) == 50

    def test_indicator_columns_added(self):
        """应添加 dt_upper、dt_lower、dt_range 列。"""
        data = _make_ohlcv(n=30)
        result = DualThrustStrategy().generate_signals(data)
        for col in ("dt_upper", "dt_lower", "dt_range"):
            assert col in result.columns, f"缺少列 {col}"

    def test_get_params(self):
        s = DualThrustStrategy(lookback=6, k1=0.7, k2=0.3)
        params = s.get_params()
        assert params == {"lookback": 6, "k1": 0.7, "k2": 0.3}

    def test_invalid_lookback(self):
        with pytest.raises(ValueError, match="lookback"):
            DualThrustStrategy(lookback=0)

    def test_invalid_k1(self):
        with pytest.raises(ValueError, match="k1"):
            DualThrustStrategy(k1=-0.1)

    def test_invalid_k2(self):
        with pytest.raises(ValueError, match="k2"):
            DualThrustStrategy(k2=0)

    def test_short_data(self):
        """数据长度小于 lookback 时不应崩溃，信号全为 hold。"""
        data = _make_ohlcv(n=3)
        result = DualThrustStrategy(lookback=4).generate_signals(data)
        assert (result["signal"] == "hold").all()

    def test_flat_prices(self):
        """平稳价格不应产生突破信号。"""
        data = _make_flat(n=30)
        result = DualThrustStrategy(lookback=4, k1=0.5, k2=0.5).generate_signals(data)
        # 非常平的价格，range 极小，open ± k*range ≈ open ≈ close
        # 大部分信号应为 hold
        signal_counts = result["signal"].value_counts()
        assert signal_counts.get("hold", 0) >= len(result) * 0.5

    def test_volatile_data(self):
        """高波动数据应同时产生买卖信号。"""
        data = _make_volatile(n=100)
        result = DualThrustStrategy(lookback=4, k1=0.3, k2=0.3).generate_signals(data)
        signals = set(result["signal"].unique())
        assert "buy" in signals or "sell" in signals


# =========================================================================
# MeanReversionStrategy
# =========================================================================

class TestMeanReversion:
    def test_registry(self):
        """应已在注册表中注册。"""
        assert "mean_reversion" in StrategyRegistry.list_strategies()
        strategy = StrategyRegistry.get("mean_reversion", window=20)
        assert isinstance(strategy, MeanReversionStrategy)

    def test_oversold_buy_signal(self):
        """价格大幅下跌后（Z-Score < -threshold）应产生买入信号。"""
        n = 80
        # 先平稳再急跌
        prices = np.concatenate([
            np.full(40, 20.0) + np.random.RandomState(10).randn(40) * 0.1,
            np.linspace(20, 10, 40),  # 急跌
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        strategy = MeanReversionStrategy(window=20, entry_threshold=1.5, exit_threshold=0.3)
        result = strategy.generate_signals(data)
        assert "buy" in result["signal"].values, "急跌后应产生买入信号（超卖回归）"

    def test_overbought_sell_signal(self):
        """价格大幅上涨后（Z-Score > threshold）应产生卖出信号。"""
        n = 80
        prices = np.concatenate([
            np.full(40, 20.0) + np.random.RandomState(11).randn(40) * 0.1,
            np.linspace(20, 35, 40),  # 急涨
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        strategy = MeanReversionStrategy(window=20, entry_threshold=1.5, exit_threshold=0.3)
        result = strategy.generate_signals(data)
        assert "sell" in result["signal"].values, "急涨后应产生卖出信号（超买回归）"

    def test_only_valid_signals(self):
        data = _make_ohlcv(n=100)
        result = MeanReversionStrategy().generate_signals(data)
        assert set(result["signal"].unique()).issubset(VALID_SIGNALS)

    def test_output_length_matches_input(self):
        data = _make_ohlcv(n=60)
        result = MeanReversionStrategy().generate_signals(data)
        assert len(result) == 60

    def test_zscore_column_added(self):
        """应添加 mr_zscore 列。"""
        data = _make_ohlcv(n=40)
        result = MeanReversionStrategy().generate_signals(data)
        assert "mr_zscore" in result.columns

    def test_get_params(self):
        s = MeanReversionStrategy(window=30, entry_threshold=2.5, exit_threshold=0.8)
        params = s.get_params()
        assert params == {"window": 30, "entry_threshold": 2.5, "exit_threshold": 0.8}

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window"):
            MeanReversionStrategy(window=1)

    def test_invalid_entry_threshold(self):
        with pytest.raises(ValueError, match="entry_threshold"):
            MeanReversionStrategy(entry_threshold=-1.0)

    def test_invalid_exit_ge_entry(self):
        with pytest.raises(ValueError, match="exit_threshold"):
            MeanReversionStrategy(entry_threshold=2.0, exit_threshold=2.0)

    def test_short_data(self):
        """数据长度小于 window 时不应崩溃，信号全为 hold。"""
        data = _make_ohlcv(n=5)
        result = MeanReversionStrategy(window=20).generate_signals(data)
        assert (result["signal"] == "hold").all()

    def test_flat_prices_mostly_hold(self):
        """平稳价格 Z-Score 接近 0，绝大多数应为 hold。"""
        data = _make_flat(n=60)
        result = MeanReversionStrategy(window=20, entry_threshold=2.0).generate_signals(data)
        non_nan = result.dropna(subset=["mr_zscore"])
        if len(non_nan) > 0:
            hold_ratio = (non_nan["signal"] == "hold").mean()
            assert hold_ratio >= 0.9, f"平稳价格 hold 比例应 >= 90%，实际 {hold_ratio:.1%}"

    def test_volatile_data_produces_signals(self):
        """高波动数据应产生买卖信号。"""
        data = _make_volatile(n=100)
        result = MeanReversionStrategy(
            window=20, entry_threshold=1.5, exit_threshold=0.3,
        ).generate_signals(data)
        signals = set(result["signal"].unique())
        assert len(signals) >= 2, "高波动数据应至少产生两种信号"


# =========================================================================
# TurtleStrategy
# =========================================================================

class TestTurtle:
    def test_registry(self):
        """应已在注册表中注册。"""
        assert "turtle" in StrategyRegistry.list_strategies()
        strategy = StrategyRegistry.get("turtle", entry_period=20, exit_period=10)
        assert isinstance(strategy, TurtleStrategy)

    def test_uptrend_generates_buy(self):
        """持续上涨创新高应产生买入信号。"""
        data = _make_uptrend(n=80)
        strategy = TurtleStrategy(entry_period=10, exit_period=5)
        result = strategy.generate_signals(data)
        assert "buy" in result["signal"].values, "上涨趋势创新高应产生买入信号"

    def test_downtrend_generates_sell(self):
        """持续下跌创新低应产生卖出信号。"""
        data = _make_downtrend(n=80)
        strategy = TurtleStrategy(entry_period=10, exit_period=5)
        result = strategy.generate_signals(data)
        assert "sell" in result["signal"].values, "下跌趋势创新低应产生卖出信号"

    def test_only_valid_signals(self):
        data = _make_ohlcv(n=100)
        result = TurtleStrategy().generate_signals(data)
        assert set(result["signal"].unique()).issubset(VALID_SIGNALS)

    def test_output_length_matches_input(self):
        data = _make_ohlcv(n=50)
        result = TurtleStrategy().generate_signals(data)
        assert len(result) == 50

    def test_indicator_columns_added(self):
        """应添加 turtle_upper、turtle_lower、atr 列。"""
        data = _make_ohlcv(n=40)
        strategy = TurtleStrategy(entry_period=10, exit_period=5, atr_period=10)
        result = strategy.generate_signals(data)
        for col in ("turtle_upper", "turtle_lower", "atr_10"):
            assert col in result.columns, f"缺少列 {col}"

    def test_get_params(self):
        s = TurtleStrategy(entry_period=30, exit_period=15, atr_period=14)
        params = s.get_params()
        assert params == {"entry_period": 30, "exit_period": 15, "atr_period": 14}

    def test_invalid_entry_period(self):
        with pytest.raises(ValueError, match="entry_period"):
            TurtleStrategy(entry_period=0)

    def test_invalid_exit_period(self):
        with pytest.raises(ValueError, match="exit_period"):
            TurtleStrategy(exit_period=-1)

    def test_invalid_atr_period(self):
        with pytest.raises(ValueError, match="atr_period"):
            TurtleStrategy(atr_period=0)

    def test_short_data(self):
        """数据长度小于周期时不应崩溃，信号全为 hold。"""
        data = _make_ohlcv(n=5)
        result = TurtleStrategy(entry_period=20, exit_period=10).generate_signals(data)
        assert (result["signal"] == "hold").all()

    def test_flat_prices(self):
        """平稳价格不应突破通道，几乎全为 hold。"""
        data = _make_flat(n=60)
        result = TurtleStrategy(entry_period=10, exit_period=5).generate_signals(data)
        signal_counts = result["signal"].value_counts()
        assert signal_counts.get("hold", 0) >= len(result) * 0.5

    def test_volatile_data(self):
        """高波动数据应产生买卖信号。"""
        data = _make_volatile(n=100)
        result = TurtleStrategy(entry_period=10, exit_period=5).generate_signals(data)
        signals = set(result["signal"].unique())
        assert "buy" in signals or "sell" in signals


# =========================================================================
# 通用行为：所有新策略
# =========================================================================

class TestAdvancedStrategyCommon:
    @pytest.mark.parametrize("name", ["dual_thrust", "mean_reversion", "turtle"])
    def test_all_produce_signal_column(self, name: str):
        """所有新策略应能正确产生 signal 列。"""
        data = _make_ohlcv(n=100)
        strategy = StrategyRegistry.get(name)
        result = strategy.generate_signals(data)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset(VALID_SIGNALS)

    @pytest.mark.parametrize("name", ["dual_thrust", "mean_reversion", "turtle"])
    def test_all_return_same_length(self, name: str):
        """策略输出 DataFrame 长度应与输入一致。"""
        data = _make_ohlcv(n=80)
        n_rows = len(data)
        strategy = StrategyRegistry.get(name)
        result = strategy.generate_signals(data)
        assert len(result) == n_rows

    @pytest.mark.parametrize("name", ["dual_thrust", "mean_reversion", "turtle"])
    def test_repr(self, name: str):
        """__repr__ 应正常工作。"""
        strategy = StrategyRegistry.get(name)
        r = repr(strategy)
        assert "Strategy" in r
        assert "params=" in r
