"""信号分析器单元测试

使用合成数据验证 SignalAnalyzer 的信号质量评估功能。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.signal_analyzer import SignalAnalysisResult, SignalAnalyzer


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_signal_data(
    n: int = 100,
    seed: int = 42,
    buy_indices: list[int] | None = None,
    sell_indices: list[int] | None = None,
) -> pd.DataFrame:
    """生成包含 close 和 signal 列的合成数据。"""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)

    signal = np.zeros(n, dtype=int)
    if buy_indices:
        for i in buy_indices:
            if i < n:
                signal[i] = 1
    if sell_indices:
        for i in sell_indices:
            if i < n:
                signal[i] = -1

    return pd.DataFrame({"close": close, "signal": signal})


def _make_trending_data_with_perfect_signals(n: int = 100) -> pd.DataFrame:
    """生成趋势数据并在上涨前发出买入信号，下跌前发出卖出信号。"""
    close = np.zeros(n)
    signal = np.zeros(n, dtype=int)

    # 构建上涨-下跌交替的行情
    close[0] = 100.0
    for i in range(1, n):
        if (i // 20) % 2 == 0:
            close[i] = close[i - 1] + 0.5  # 上涨段
        else:
            close[i] = close[i - 1] - 0.5  # 下跌段

    # 在上涨段起点发买入信号，下跌段起点发卖出信号
    for i in range(0, n, 20):
        segment = (i // 20) % 2
        if segment == 0 and i < n:
            signal[i] = 1  # 上涨段起点买入
        elif segment == 1 and i < n:
            signal[i] = -1  # 下跌段起点卖出

    return pd.DataFrame({"close": close, "signal": signal})


# =========================================================================
# SignalAnalyzer.analyze
# =========================================================================

class TestSignalAnalyzerAnalyze:
    def test_basic_analysis(self):
        """基本分析：验证返回的信号数量统计。"""
        data = _make_signal_data(
            n=100,
            buy_indices=[5, 25, 45, 65, 85],
            sell_indices=[15, 35, 55, 75],
        )
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        assert isinstance(result, SignalAnalysisResult)
        assert result.buy_signals == 5
        assert result.sell_signals == 4
        assert result.total_signals == 9

    def test_signal_frequency(self):
        """信号频率 = 信号总数 / 总天数。"""
        data = _make_signal_data(
            n=100,
            buy_indices=[10, 50],
            sell_indices=[30, 70],
        )
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        expected_freq = 4 / 100
        assert abs(result.signal_frequency - expected_freq) < 1e-10

    def test_no_signals(self):
        """无信号时应返回零值结果。"""
        data = _make_signal_data(n=50)
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        assert result.total_signals == 0
        assert result.buy_signals == 0
        assert result.sell_signals == 0
        assert result.avg_return_after_buy == 0.0
        assert result.avg_return_after_sell == 0.0
        assert result.buy_win_rate == 0.0
        assert result.sell_win_rate == 0.0
        assert result.consecutive_wins == 0
        assert result.consecutive_losses == 0

    def test_only_buy_signals(self):
        """仅有买入信号时卖出相关指标应为0。"""
        data = _make_signal_data(n=50, buy_indices=[5, 15, 25])
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        assert result.buy_signals == 3
        assert result.sell_signals == 0
        assert result.avg_return_after_sell == 0.0
        assert result.sell_win_rate == 0.0

    def test_only_sell_signals(self):
        """仅有卖出信号时买入相关指标应为0。"""
        data = _make_signal_data(n=50, sell_indices=[5, 15, 25])
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        assert result.sell_signals == 3
        assert result.buy_signals == 0
        assert result.avg_return_after_buy == 0.0
        assert result.buy_win_rate == 0.0


# =========================================================================
# 胜率计算
# =========================================================================

class TestWinRate:
    def test_perfect_buy_signals(self):
        """在已知上涨序列中的买入信号胜率应为100%。"""
        # 创建单调上涨序列
        n = 50
        close = np.arange(100, 100 + n, dtype=float)
        signal = np.zeros(n, dtype=int)
        signal[5] = 1
        signal[15] = 1
        signal[25] = 1
        data = pd.DataFrame({"close": close, "signal": signal})

        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, forward_periods=[1])

        assert result.buy_win_rate == 1.0

    def test_perfect_sell_signals_in_downtrend(self):
        """在已知下跌序列中的卖出信号胜率应为100%。"""
        n = 50
        close = np.arange(200, 200 - n, -1, dtype=float)
        signal = np.zeros(n, dtype=int)
        signal[5] = -1
        signal[15] = -1
        signal[25] = -1
        data = pd.DataFrame({"close": close, "signal": signal})

        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, forward_periods=[1])

        assert result.sell_win_rate == 1.0

    def test_mixed_win_rate(self):
        """验证已知盈亏比例。"""
        # 手工构造：买入后有2次上涨、1次下跌
        close = [100, 101, 102, 100, 99, 98, 100, 101, 103, 104, 105]
        signal = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        data = pd.DataFrame({"close": close, "signal": signal})

        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, forward_periods=[1])

        # idx=1: close=101->102, 涨 (win)
        # idx=4: close=99->98, 跌 (loss)
        # idx=7: close=101->103, 涨 (win)
        assert result.buy_signals == 3
        assert abs(result.buy_win_rate - 2 / 3) < 1e-10


# =========================================================================
# 前瞻收益分析
# =========================================================================

class TestForwardReturn:
    def test_forward_periods_parameter(self):
        """自定义 forward_periods 不应报错。"""
        data = _make_signal_data(n=100, buy_indices=[5, 25])
        analyzer = SignalAnalyzer()

        # 不同的前瞻周期
        result_1 = analyzer.analyze(data, forward_periods=[1])
        result_5 = analyzer.analyze(data, forward_periods=[5])

        # 不同周期应（通常）产生不同的平均收益
        assert isinstance(result_1.avg_return_after_buy, float)
        assert isinstance(result_5.avg_return_after_buy, float)

    def test_known_forward_return(self):
        """已知收益率的前瞻收益验证。"""
        # close: [100, 110, 120, ...] -> 1日收益率 = 10%
        close = [100.0 * (1.1 ** i) for i in range(20)]
        signal = [0] * 20
        signal[0] = 1
        signal[5] = 1
        data = pd.DataFrame({"close": close, "signal": signal})

        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, forward_periods=[1])

        # 每日收益 ≈ 10%
        assert abs(result.avg_return_after_buy - 0.10) < 0.01


# =========================================================================
# 平均持仓天数
# =========================================================================

class TestHoldingPeriod:
    def test_known_holding_period(self):
        """已知买卖间隔的持仓天数验证。"""
        data = _make_signal_data(
            n=50,
            buy_indices=[5, 25],
            sell_indices=[15, 35],
        )
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        # 两次持仓：5->15 (10天), 25->35 (10天)
        assert result.avg_holding_period == 10.0

    def test_no_matching_sell(self):
        """无对应卖出信号时持仓天数为0。"""
        data = _make_signal_data(n=50, buy_indices=[5, 25])
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        assert result.avg_holding_period == 0.0

    def test_uneven_holding_periods(self):
        """不均匀持仓天数的平均值。"""
        data = _make_signal_data(
            n=50,
            buy_indices=[5, 20],
            sell_indices=[10, 40],
        )
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)

        # 5->10 (5天), 20->40 (20天), 平均 12.5
        assert abs(result.avg_holding_period - 12.5) < 1e-10


# =========================================================================
# 连胜/连亏
# =========================================================================

class TestConsecutiveStreaks:
    def test_all_wins(self):
        """全部盈利的连胜。"""
        n = 50
        close = np.arange(100, 100 + n, dtype=float)
        signal = np.zeros(n, dtype=int)
        signal[0] = 1
        signal[10] = 1
        signal[20] = 1
        data = pd.DataFrame({"close": close, "signal": signal})

        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, forward_periods=[1])

        assert result.consecutive_wins == 3
        assert result.consecutive_losses == 0

    def test_all_losses(self):
        """全部亏损的连亏。"""
        n = 50
        close = np.arange(200, 200 - n, -1, dtype=float)
        signal = np.zeros(n, dtype=int)
        signal[0] = 1  # 买入但后续下跌
        signal[10] = 1
        signal[20] = 1
        data = pd.DataFrame({"close": close, "signal": signal})

        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, forward_periods=[1])

        assert result.consecutive_losses == 3
        assert result.consecutive_wins == 0


# =========================================================================
# 边界情况
# =========================================================================

class TestEdgeCases:
    def test_empty_dataframe_raises(self):
        """空DataFrame应抛出异常。"""
        data = pd.DataFrame(columns=["close", "signal"])
        analyzer = SignalAnalyzer()
        with pytest.raises(ValueError, match="输入数据为空"):
            analyzer.analyze(data)

    def test_missing_columns_raises(self):
        """缺少必需列应抛出异常。"""
        data = pd.DataFrame({"price": [1, 2, 3]})
        analyzer = SignalAnalyzer()
        with pytest.raises(ValueError, match="缺少必需列"):
            analyzer.analyze(data)

    def test_single_row(self):
        """单行数据不应崩溃。"""
        data = pd.DataFrame({"close": [100.0], "signal": [1]})
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)
        assert result.total_signals == 1

    def test_all_signals_at_end(self):
        """所有信号在末尾（无有效前瞻收益）。"""
        n = 10
        data = _make_signal_data(n=n, buy_indices=[8, 9])
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, forward_periods=[5])
        # 末尾信号的前瞻收益为 NaN，不计入统计
        assert result.buy_win_rate == 0.0

    def test_custom_signal_column(self):
        """自定义信号列名。"""
        data = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "my_signal": [1, 0, 0, -1, 0],
        })
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data, signal_column="my_signal")
        assert result.buy_signals == 1
        assert result.sell_signals == 1


# =========================================================================
# summary 输出
# =========================================================================

class TestSummary:
    def test_summary_format(self):
        """summary() 应返回非空字符串。"""
        data = _make_signal_data(
            n=100, buy_indices=[10, 30, 50], sell_indices=[20, 40, 60]
        )
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(data)
        summary = result.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "信号质量分析报告" in summary
        assert "信号总数" in summary
