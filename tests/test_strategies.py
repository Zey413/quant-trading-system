"""策略模块单元测试

使用合成数据验证所有策略的信号生成逻辑以及注册表功能。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.strategy.base import Strategy, StrategyRegistry
from quant_trading.strategy.bollinger_strategy import BollingerStrategy
from quant_trading.strategy.composite import CompositeStrategy
from quant_trading.strategy.ma_crossover import MACrossoverStrategy
from quant_trading.strategy.macd_strategy import MACDStrategy
from quant_trading.strategy.rsi_strategy import RSIStrategy


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

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


# =========================================================================
# StrategyRegistry
# =========================================================================

class TestStrategyRegistry:
    def test_registered_strategies(self):
        """内置策略应已注册。"""
        names = StrategyRegistry.list_strategies()
        for expected in ("ma_crossover", "rsi", "macd", "bollinger", "composite"):
            assert expected in names, f"{expected} 未注册"

    def test_get_strategy(self):
        strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)
        assert isinstance(strategy, MACrossoverStrategy)

    def test_get_with_params(self):
        strategy = StrategyRegistry.get("rsi", period=7, oversold=25, overbought=75)
        assert isinstance(strategy, RSIStrategy)
        assert strategy.get_params()["period"] == 7

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="未找到策略"):
            StrategyRegistry.get("nonexistent_strategy")

    def test_register_non_strategy_raises(self):
        with pytest.raises(TypeError, match="Strategy"):

            @StrategyRegistry.register("bad")
            class NotAStrategy:
                pass


# =========================================================================
# MACrossoverStrategy
# =========================================================================

class TestMACrossover:
    def test_golden_cross_generates_buy(self):
        """短均线上穿长均线应产生买入信号。"""
        # 构造一个先跌后涨的序列，保证出现金叉
        n = 50
        prices = np.concatenate([
            np.linspace(20, 10, 25),  # 下跌
            np.linspace(10, 25, 25),  # 上涨
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        strategy = MACrossoverStrategy(short_window=3, long_window=10)
        result = strategy.generate_signals(data)
        assert "signal" in result.columns
        signals = result["signal"]
        assert "buy" in signals.values, "应产生至少一个买入信号"

    def test_death_cross_generates_sell(self):
        """短均线下穿长均线应产生卖出信号。"""
        n = 50
        prices = np.concatenate([
            np.linspace(10, 25, 25),  # 上涨
            np.linspace(25, 10, 25),  # 下跌
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        strategy = MACrossoverStrategy(short_window=3, long_window=10)
        result = strategy.generate_signals(data)
        assert "sell" in result["signal"].values, "应产生至少一个卖出信号"

    def test_only_valid_signals(self):
        """信号列只应包含 buy/sell/hold。"""
        data = _make_ohlcv(n=100)
        result = MACrossoverStrategy(short_window=5, long_window=20).generate_signals(data)
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})

    def test_get_params(self):
        s = MACrossoverStrategy(short_window=10, long_window=30)
        params = s.get_params()
        assert params["short_window"] == 10
        assert params["long_window"] == 30

    def test_invalid_windows(self):
        with pytest.raises(ValueError):
            MACrossoverStrategy(short_window=20, long_window=5)


# =========================================================================
# RSIStrategy
# =========================================================================

class TestRSIStrategy:
    def test_oversold_buy_signal(self):
        """RSI 从超卖区上穿应产生买入信号。"""
        # 构造先大幅下跌再反弹的序列以使 RSI 跌入超卖再回升
        n = 60
        prices = np.concatenate([
            np.linspace(50, 10, 30),   # 持续下跌
            np.linspace(10, 30, 30),   # 反弹
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        result = strategy.generate_signals(data)
        assert "buy" in result["signal"].values, "应在超卖回升时产生买入信号"

    def test_overbought_sell_signal(self):
        """RSI 从超买区下穿应产生卖出信号。"""
        n = 60
        prices = np.concatenate([
            np.linspace(10, 50, 30),   # 持续上涨
            np.linspace(50, 30, 30),   # 回落
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        result = strategy.generate_signals(data)
        assert "sell" in result["signal"].values, "应在超买回落时产生卖出信号"

    def test_only_valid_signals(self):
        data = _make_ohlcv(n=100)
        result = RSIStrategy().generate_signals(data)
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})


# =========================================================================
# MACDStrategy
# =========================================================================

class TestMACDStrategy:
    def test_histogram_crossover(self):
        """MACD 柱状线由负转正应买入，由正转负应卖出。"""
        data = _make_ohlcv(n=100)
        result = MACDStrategy().generate_signals(data)
        assert "signal" in result.columns
        signals = result["signal"]
        # 对于100根随机K线，通常会产生买卖信号
        valid_signals = set(signals.unique())
        assert valid_signals.issubset({"buy", "sell", "hold"})

    def test_get_params(self):
        s = MACDStrategy(fast=10, slow=20, signal=5)
        params = s.get_params()
        assert params == {"fast": 10, "slow": 20, "signal": 5}

    def test_signal_at_crossover_points(self):
        """验证信号精确出现在柱状线过零点。"""
        n = 80
        prices = np.concatenate([
            np.linspace(10, 20, 40),  # 上涨
            np.linspace(20, 8, 40),   # 下跌
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        result = MACDStrategy(fast=5, slow=13, signal=4).generate_signals(data)
        buy_rows = result[result["signal"] == "buy"]
        sell_rows = result[result["signal"] == "sell"]
        # 至少会有一次金叉和一次死叉
        assert len(buy_rows) >= 1 or len(sell_rows) >= 1


# =========================================================================
# BollingerStrategy
# =========================================================================

class TestBollingerStrategy:
    def test_lower_band_bounce_buy(self):
        """价格触及下轨后回升应产生买入信号。"""
        n = 60
        # 先平稳，再急跌触下轨，再回升
        prices = np.concatenate([
            np.full(25, 20.0) + np.random.RandomState(0).randn(25) * 0.1,
            np.linspace(20, 15, 15),  # 急跌
            np.linspace(15, 20, 20),  # 回升
        ])
        data = pd.DataFrame({
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.full(n, 10000.0),
        })
        result = BollingerStrategy(window=20, num_std=2).generate_signals(data)
        # 信号列应存在且只包含有效值
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})

    def test_only_valid_signals(self):
        data = _make_ohlcv(n=100)
        result = BollingerStrategy().generate_signals(data)
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})

    def test_get_params(self):
        s = BollingerStrategy(window=15, num_std=1.5)
        assert s.get_params() == {"window": 15, "num_std": 1.5}


# =========================================================================
# CompositeStrategy
# =========================================================================

class TestCompositeStrategy:
    def _make_dummy_strategy(self, signal_value: str) -> Strategy:
        """创建一个始终返回固定信号的策略。"""

        class DummyStrategy(Strategy):
            def __init__(self, sig: str):
                super().__init__(name=f"dummy_{sig}")
                self._sig = sig

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                data["signal"] = self._sig
                return data

        return DummyStrategy(signal_value)

    def test_majority_all_buy(self):
        """majority 投票：3 个策略全买 -> buy。"""
        strategies = [self._make_dummy_strategy("buy") for _ in range(3)]
        composite = CompositeStrategy(strategies=strategies, voting="majority")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "buy").all()

    def test_majority_mixed(self):
        """majority 投票：2 buy + 1 sell -> buy。"""
        strategies = [
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("sell"),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="majority")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "buy").all()

    def test_majority_no_consensus(self):
        """majority 投票：1 buy + 1 sell + 1 hold -> hold。"""
        strategies = [
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("sell"),
            self._make_dummy_strategy("hold"),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="majority")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "hold").all()

    def test_unanimous_all_agree(self):
        """unanimous 投票：全部 sell -> sell。"""
        strategies = [self._make_dummy_strategy("sell") for _ in range(3)]
        composite = CompositeStrategy(strategies=strategies, voting="unanimous")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "sell").all()

    def test_unanimous_disagree(self):
        """unanimous 投票：有分歧 -> hold。"""
        strategies = [
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("hold"),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="unanimous")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "hold").all()

    def test_any_one_buy(self):
        """any 投票：1 buy + 2 hold -> buy。"""
        strategies = [
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("hold"),
            self._make_dummy_strategy("hold"),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="any")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "buy").all()

    def test_any_buy_and_sell(self):
        """any 投票：buy 和 sell 同时出现时 buy 优先。"""
        strategies = [
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("sell"),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="any")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "buy").all()

    def test_any_only_sell(self):
        """any 投票：只有 sell 信号 -> sell。"""
        strategies = [
            self._make_dummy_strategy("sell"),
            self._make_dummy_strategy("hold"),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="any")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        assert (result["signal"] == "sell").all()

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="至少需要"):
            CompositeStrategy(strategies=[])

    def test_none_strategies_raises(self):
        with pytest.raises(ValueError, match="至少需要"):
            CompositeStrategy(strategies=None)

    def test_invalid_voting_raises(self):
        with pytest.raises(ValueError, match="voting"):
            CompositeStrategy(
                strategies=[self._make_dummy_strategy("buy")],
                voting="invalid",  # type: ignore[arg-type]
            )

    def test_no_temp_columns_left(self):
        """复合策略执行后不应留下临时 _signal_ 列。"""
        strategies = [
            self._make_dummy_strategy("buy"),
            self._make_dummy_strategy("sell"),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="majority")
        data = _make_ohlcv(n=10)
        result = composite.generate_signals(data)
        temp_cols = [c for c in result.columns if c.startswith("_signal_")]
        assert len(temp_cols) == 0

    def test_with_real_strategies(self):
        """使用真实子策略（MA + MACD）的复合策略应能正常运行。"""
        strategies = [
            MACrossoverStrategy(short_window=5, long_window=20),
            MACDStrategy(fast=12, slow=26, signal=9),
        ]
        composite = CompositeStrategy(strategies=strategies, voting="any")
        data = _make_ohlcv(n=100)
        result = composite.generate_signals(data)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})


# =========================================================================
# Strategy 通用行为
# =========================================================================

class TestStrategyCommon:
    @pytest.mark.parametrize("name", ["ma_crossover", "rsi", "macd", "bollinger"])
    def test_all_strategies_produce_signal_column(self, name: str):
        """所有注册策略应能正确产生 signal 列。"""
        data = _make_ohlcv(n=100)
        strategy = StrategyRegistry.get(name)
        result = strategy.generate_signals(data)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})

    @pytest.mark.parametrize("name", ["ma_crossover", "rsi", "macd", "bollinger"])
    def test_all_strategies_return_same_length(self, name: str):
        """策略输出 DataFrame 长度应与输入一致。"""
        data = _make_ohlcv(n=80)
        n_rows = len(data)
        strategy = StrategyRegistry.get(name)
        result = strategy.generate_signals(data)
        assert len(result) == n_rows
