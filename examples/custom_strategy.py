"""
自定义策略示例 - 展示如何创建自己的交易策略
=============================================

本示例实现一个 "双均线 + 成交量确认" 策略:

买入条件 (两个条件同时满足):
  1. 短期均线上穿长期均线（金叉）
  2. 当日成交量 > N日平均成交量（放量确认）

卖出条件:
  - 短期均线下穿长期均线（死叉）

核心知识点:
  1. 继承 Strategy 基类，实现 generate_signals() 方法
  2. 使用 @StrategyRegistry.register(name) 装饰器注册策略
  3. 使用内置技术指标 (SMA) 简化计算
  4. 使用 BacktestEngine 运行回测
  5. 使用 ChartGenerator 可视化结果

运行方式:
    cd <项目根目录>
    python examples/custom_strategy.py

策略思路:
    传统的均线交叉策略容易在震荡市中产生大量假信号。
    加入成交量确认条件后，可以过滤掉一部分缩量的假金叉，
    从而提高信号质量。放量金叉通常意味着更多资金入场，
    趋势转折的可靠性更高。
"""

import sys
from pathlib import Path

import pandas as pd

# ================================================================== #
#  确保项目路径在 sys.path 中 (方便直接运行脚本)
# ================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ================================================================== #
#  导入框架组件
# ================================================================== #
from quant_trading.core.config import AppConfig
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.data.manager import DataManager
from quant_trading.indicators.trend import SMA
from quant_trading.strategy.base import Strategy, StrategyRegistry


# ================================================================== #
#  第1步: 定义自定义策略
#
#  所有策略必须:
#  1. 继承 Strategy 基类
#  2. 实现 generate_signals(data) 方法
#  3. generate_signals() 返回的 DataFrame 必须包含 "signal" 列
#     (值为 "buy" / "sell" / "hold")
#
#  可选:
#  - 使用 @StrategyRegistry.register(name) 装饰器注册到注册表
#  - 重写 get_params() 方法返回策略参数字典
#  - 在 __init__() 中接受自定义参数
# ================================================================== #

@StrategyRegistry.register("ma_volume")
class MAVolumeStrategy(Strategy):
    """双均线 + 成交量确认策略。

    融合了趋势跟踪（均线交叉）和量价配合（放量确认）两个维度，
    旨在过滤震荡市中的假信号，提高信号质量。

    买入条件 (同时满足):
        1. 短期均线上穿长期均线 (金叉)
        2. 当日成交量 > N日平均成交量 (放量确认)

    卖出条件:
        - 短期均线下穿长期均线 (死叉)

    Parameters
    ----------
    short_window : int
        短期均线周期，默认 5 (一周)
    long_window : int
        长期均线周期，默认 20 (一个月)
    volume_window : int
        成交量均线周期，默认 20
    volume_ratio : float
        成交量需达到均量的倍数，默认 1.0 (即超过均量即可)
        设为 1.5 则要求放量50%以上
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        volume_window: int = 20,
        volume_ratio: float = 1.0,
        name: str = "",
    ) -> None:
        # 调用父类构造函数（设置策略名称）
        super().__init__(name=name or "ma_volume")

        # 参数校验
        if short_window >= long_window:
            raise ValueError(
                f"短期均线周期 ({short_window}) 必须小于长期均线周期 ({long_window})"
            )

        # 保存策略参数
        self.short_window = short_window
        self.long_window = long_window
        self.volume_window = volume_window
        self.volume_ratio = volume_ratio

        # 创建指标实例 (使用框架内置的 SMA 指标)
        # 也可以直接用 pandas 的 rolling().mean() 计算，但使用指标类更规范
        self._sma_short = SMA(window=short_window)
        self._sma_long = SMA(window=long_window)

    def get_params(self) -> dict:
        """返回策略参数字典 (用于日志、序列化、CLI展示)"""
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "volume_window": self.volume_window,
            "volume_ratio": self.volume_ratio,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号。

        这是策略的核心方法，接收标准化的 OHLCV 数据，返回添加了
        signal 列和指标列的 DataFrame。

        Parameters
        ----------
        data : pd.DataFrame
            标准化行情数据，包含 date/open/high/low/close/volume 列

        Returns
        -------
        pd.DataFrame
            添加了以下列的 DataFrame:
            - sma_{short_window}: 短期均线
            - sma_{long_window}: 长期均线
            - vol_ma: 成交量均线
            - signal: 交易信号 ("buy"/"sell"/"hold")
        """
        # 注意: 对传入的数据做一份拷贝，避免修改原始数据
        df = data.copy()

        # ---- 计算技术指标 ----

        # 使用内置 SMA 指标计算均线
        # calculate() 方法会在 DataFrame 中添加 sma_{window} 列
        short_col = f"sma_{self.short_window}"
        long_col = f"sma_{self.long_window}"

        if short_col not in df.columns:
            df = self._sma_short.calculate(df)
        if long_col not in df.columns:
            df = self._sma_long.calculate(df)

        # 成交量均线 (直接用 pandas 计算)
        df["vol_ma"] = df["volume"].rolling(
            window=self.volume_window, min_periods=self.volume_window
        ).mean()

        # ---- 生成信号 ----

        # 获取均线序列
        short_ma = df[short_col]
        long_ma = df[long_col]

        # 前一个交易日的均线值 (用于判断交叉)
        prev_short = short_ma.shift(1)
        prev_long = long_ma.shift(1)

        # 条件1: 金叉 — 前一日短均线 <= 长均线，今日短均线 > 长均线
        golden_cross = (prev_short <= prev_long) & (short_ma > long_ma)

        # 条件2: 放量确认 — 当日成交量 > 均量 × 倍数
        volume_confirm = df["volume"] > (df["vol_ma"] * self.volume_ratio)

        # 买入信号: 金叉 + 放量 (两个条件同时满足)
        buy_signal = golden_cross & volume_confirm

        # 卖出信号: 死叉 — 前一日短均线 >= 长均线，今日短均线 < 长均线
        # 死叉不需要成交量确认 (止损/止盈不应该有额外条件)
        death_cross = (prev_short >= prev_long) & (short_ma < long_ma)

        # 初始化信号列
        df["signal"] = "hold"
        df.loc[buy_signal, "signal"] = "buy"
        df.loc[death_cross, "signal"] = "sell"

        return df


# ================================================================== #
#  第2步: 运行回测并展示结果
# ================================================================== #

def main():
    print("=" * 60)
    print("  自定义策略示例 - 双均线 + 成交量确认")
    print("=" * 60)

    # ---- 加载配置 ----
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = AppConfig.from_yaml(str(config_path))

    # ---- 方法1: 直接实例化 ----
    # 可以直接创建策略实例，传入自定义参数
    strategy = MAVolumeStrategy(
        short_window=5,
        long_window=20,
        volume_window=20,
        volume_ratio=1.0,   # 成交量超过均量即可
    )
    print(f"\n🎯 自定义策略: {strategy.name}")
    print(f"   参数: {strategy.get_params()}")

    # ---- 方法2: 通过注册表获取 ----
    # 因为用了 @StrategyRegistry.register("ma_volume")，
    # 也可以通过注册表获取（方便CLI和配置化使用）
    strategy_from_registry = StrategyRegistry.get("ma_volume")
    print(f"\n📋 从注册表获取: {strategy_from_registry.name}")
    print(f"   可用策略列表: {StrategyRegistry.list_strategies()}")

    # ---- 获取数据 ----
    dm = DataManager(config)
    symbol = "000001"  # 平安银行
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    print(f"\n📊 正在获取 {symbol} 行情数据...")
    try:
        data = dm.fetch_daily(symbol, start_date, end_date)
        print(f"✅ 获取到 {len(data)} 条数据")
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        print("   请确保已安装 akshare: pip install akshare")
        return

    if data.empty:
        print("❌ 未获取到数据，请检查股票代码和日期范围")
        return

    # ---- 预览信号 ----
    # 在运行回测之前，可以先看看策略产生了哪些信号
    signal_data = strategy.generate_signals(data.copy())
    buy_signals = signal_data[signal_data['signal'] == 'buy']
    sell_signals = signal_data[signal_data['signal'] == 'sell']

    print(f"\n📡 信号预览:")
    print(f"   买入信号: {len(buy_signals)} 次")
    print(f"   卖出信号: {len(sell_signals)} 次")

    if not buy_signals.empty:
        print(f"\n   买入信号详情:")
        for _, row in buy_signals.iterrows():
            date_str = (row['date'].strftime('%Y-%m-%d')
                        if hasattr(row['date'], 'strftime') else str(row['date']))
            print(f"     {date_str}  收盘: {row['close']:.2f}  "
                  f"成交量: {int(row['volume']):,}")

    # ---- 运行回测 ----
    print(f"\n⏳ 正在运行回测...")
    engine = BacktestEngine(config)
    result = engine.run(strategy, data, symbol=symbol)

    # ---- 打印绩效报告 ----
    print(result.summary())

    # ---- 与原始均线策略对比 ----
    print("\n" + "=" * 60)
    print("  策略对比: 双均线+放量 vs 纯双均线")
    print("=" * 60)

    # 运行纯均线策略作为对照
    from quant_trading.strategy.ma_crossover import MACrossoverStrategy
    baseline = MACrossoverStrategy(short_window=5, long_window=20)
    baseline_result = engine.run(baseline, data, symbol=symbol)

    print(f"\n{'指标':<16} {'双均线+放量':>14} {'纯双均线':>14}")
    print(f"{'-' * 16} {'-' * 14} {'-' * 14}")
    print(f"{'总收益率':<16} {result.total_return:>+13.2%} {baseline_result.total_return:>+13.2%}")
    print(f"{'年化收益率':<14} {result.annualized_return:>+13.2%} {baseline_result.annualized_return:>+13.2%}")
    print(f"{'最大回撤':<16} {result.max_drawdown:>13.2%} {baseline_result.max_drawdown:>13.2%}")
    print(f"{'夏普比率':<16} {result.sharpe_ratio:>13.4f} {baseline_result.sharpe_ratio:>13.4f}")
    print(f"{'胜率':<18} {result.win_rate:>13.2%} {baseline_result.win_rate:>13.2%}")
    print(f"{'交易次数':<16} {result.total_trades:>13d} {baseline_result.total_trades:>13d}")

    # ---- 可视化 ----
    print("\n📈 正在生成图表...")
    try:
        from quant_trading.visualization.charts import ChartGenerator

        # 交易信号图 — 可以直观看到买卖点的位置
        ChartGenerator.plot_signals(
            signal_data,
            title=f"{symbol} 双均线+成交量确认策略",
            # save_path="output/ma_volume_signals.png",
        )

        # 净值曲线对比
        if not result.equity_curve.empty and not baseline_result.equity_curve.empty:
            # 将两条净值曲线归一化到起始值为1
            norm_strategy = result.equity_curve / result.equity_curve.iloc[0]
            norm_baseline = baseline_result.equity_curve / baseline_result.equity_curve.iloc[0]

            ChartGenerator.plot_equity_curve(
                norm_strategy,
                benchmark=norm_baseline,
                title=f"{symbol} 策略净值对比 (蓝=自定义策略, 橙=纯均线)",
                # save_path="output/ma_volume_equity.png",
            )

        print("✅ 图表生成完成!")

    except ImportError as e:
        print(f"⚠️  图表生成跳过 (缺少依赖): {e}")
    except Exception as e:
        print(f"⚠️  图表生成失败: {e}")

    # ============================================================== #
    #  完成!
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  自定义策略示例运行完成!")
    print("")
    print("  进阶练习:")
    print("    • 调整 volume_ratio 参数 (如 1.5)，观察信号变化")
    print("    • 添加 RSI 过滤条件 (RSI < 30 时买入)")
    print("    • 使用 CompositeStrategy 组合多个策略")
    print("    • 对多只股票进行回测 (engine.run_multi)")
    print("=" * 60)


if __name__ == "__main__":
    main()
