"""
快速入门示例 - 量化交易系统
============================

本示例演示如何使用量化交易系统的核心功能:
1. 加载配置
2. 获取A股行情数据 (以平安银行 000001 为例)
3. 运行均线交叉策略回测
4. 打印绩效摘要
5. 生成可视化图表

运行方式:
    cd <项目根目录>
    pip install -e ".[dev]"          # 首次运行需安装
    python examples/quick_start.py

注意:
    - 首次运行需要网络连接 (从akshare获取数据)
    - 数据会被缓存至 data_cache/ 目录，后续运行速度更快
    - 图表会弹出matplotlib窗口，关闭窗口后程序继续执行
    - 如需保存图表为文件，取消对应代码行的注释即可
"""

import sys
from pathlib import Path

# ================================================================== #
#  第0步: 确保项目路径在 sys.path 中
#  (方便直接 python examples/quick_start.py 运行，无需安装)
# ================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    # ============================================================== #
    #  第1步: 加载配置
    #
    #  AppConfig 使用 Pydantic 管理配置，可从 YAML 文件加载。
    #  配置包含: 数据源设置、券商参数(佣金/印花税)、回测参数、风控参数
    # ============================================================== #
    print("=" * 60)
    print("  量化交易系统 - 快速入门示例")
    print("=" * 60)

    from quant_trading.core.config import AppConfig

    # 从项目根目录下的配置文件加载（文件不存在时使用默认值）
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = AppConfig.from_yaml(str(config_path))

    print(f"\n✅ 配置加载完成")
    print(f"   数据源:     {config.data_source.default_source}")
    print(f"   初始资金:   {config.backtest.initial_capital:,.0f} 元")
    print(f"   回测区间:   {config.backtest.start_date} ~ {config.backtest.end_date}")
    print(f"   佣金率:     {config.broker.commission_rate * 10000:.1f}‱ (万分之三)")
    print(f"   印花税率:   {config.broker.stamp_tax_rate * 1000:.1f}‰ (仅卖出)")
    print(f"   T+1规则:    {'开启' if config.broker.t_plus_1 else '关闭'}")

    # ============================================================== #
    #  第2步: 获取行情数据
    #
    #  DataManager 是数据层的门面(Facade):
    #  - 自动管理数据源切换 (akshare/tushare)
    #  - 内置 Parquet 文件缓存
    #  - 返回标准化 DataFrame (date/open/high/low/close/volume)
    # ============================================================== #
    from quant_trading.data.manager import DataManager

    dm = DataManager(config)

    # 平安银行 (000001) — A股最经典的标的之一
    # 你可以改成其他股票代码，比如:
    #   600519 贵州茅台
    #   000858 五粮液
    #   601318 中国平安
    symbol = "000001"
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    print(f"\n📊 正在获取 {symbol} 行情数据 ({start_date} ~ {end_date})...")

    try:
        data = dm.fetch_daily(symbol, start_date, end_date)
    except Exception as e:
        print(f"\n❌ 数据获取失败: {e}")
        print("   请检查网络连接，或确保已安装 akshare:")
        print("   pip install akshare")
        return

    if data.empty:
        print("❌ 未获取到数据，请检查股票代码和日期范围")
        return

    print(f"✅ 获取到 {len(data)} 条日线数据")
    print(f"   日期范围: {data['date'].iloc[0]} ~ {data['date'].iloc[-1]}")

    # 展示数据预览
    print(f"\n   数据预览 (前5条):")
    print(f"   {'日期':^12} {'开盘':>8} {'最高':>8} {'最低':>8} {'收盘':>8} {'成交量':>12}")
    print(f"   {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}")
    for _, row in data.head(5).iterrows():
        date_str = (row['date'].strftime('%Y-%m-%d')
                    if hasattr(row['date'], 'strftime') else str(row['date']))
        print(f"   {date_str:^12} {row['open']:>8.2f} {row['high']:>8.2f} "
              f"{row['low']:>8.2f} {row['close']:>8.2f} {int(row['volume']):>12,}")

    # ============================================================== #
    #  第3步: 运行均线交叉策略回测
    #
    #  StrategyRegistry 是策略注册表:
    #  - 内置策略在导入时自动注册 (ma_crossover, rsi, macd, bollinger, composite)
    #  - 通过 StrategyRegistry.get(name, **params) 获取策略实例
    #
    #  BacktestEngine 是事件驱动回测引擎:
    #  - 逐日模拟交易过程
    #  - 自动处理佣金/印花税/滑点/T+1等A股规则
    #  - 返回 PerformanceResult 绩效结果
    # ============================================================== #
    # 导入策略模块 (会自动触发所有内置策略的注册)
    from quant_trading.strategy import StrategyRegistry
    from quant_trading.backtest.engine import BacktestEngine

    # 查看所有可用策略
    available_strategies = StrategyRegistry.list_strategies()
    print(f"\n📋 可用策略: {available_strategies}")

    # 创建均线交叉策略实例
    # 参数说明:
    #   short_window=5  -> 短期均线: 5日均线
    #   long_window=20  -> 长期均线: 20日均线
    # 信号规则:
    #   金叉 (5日均线上穿20日均线) -> 买入
    #   死叉 (5日均线下穿20日均线) -> 卖出
    strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)
    print(f"\n🎯 当前策略: {strategy.name}")
    print(f"   参数: {strategy.get_params()}")

    # 创建回测引擎并运行
    engine = BacktestEngine(config)
    print(f"\n⏳ 正在运行回测 (初始资金: {config.backtest.initial_capital:,.0f}元)...")
    result = engine.run(strategy, data, symbol=symbol)

    # ============================================================== #
    #  第4步: 打印绩效摘要
    #
    #  PerformanceResult 包含:
    #  - 收益指标: 总收益率、年化收益率
    #  - 风险指标: 最大回撤、波动率
    #  - 风险调整收益: 夏普比率、Sortino比率、Calmar比率
    #  - 交易统计: 胜率、盈亏比
    #  - 时间序列: equity_curve (净值曲线)、drawdown_series (回撤序列)
    # ============================================================== #
    print(result.summary())

    # 也可以单独访问各个指标
    print("📊 关键指标一览:")
    print(f"   总收益率:      {result.total_return:>+8.2%}")
    print(f"   年化收益率:    {result.annualized_return:>+8.2%}")
    print(f"   最大回撤:      {result.max_drawdown:>8.2%}")
    print(f"   夏普比率:      {result.sharpe_ratio:>8.4f}")
    print(f"   Sortino比率:   {result.sortino_ratio:>8.4f}")
    print(f"   胜率:          {result.win_rate:>8.2%}")
    print(f"   总交易次数:    {result.total_trades:>8d}")

    # ============================================================== #
    #  第5步: 生成可视化图表
    #
    #  ChartGenerator 提供多种静态方法:
    #  - plot_candlestick()    K线图 (红涨绿跌)
    #  - plot_signals()        交易信号图 (买卖点标注)
    #  - plot_equity_curve()   净值曲线图 (含回撤填充)
    #  - plot_drawdown()       回撤图
    #  - plot_monthly_returns() 月度收益热力图
    #  - plot_backtest_report() 完整回测报告 (四合一)
    #
    #  每个方法都接受 save_path 参数:
    #  - None -> 弹出matplotlib窗口显示
    #  - "path/to/file.png" -> 保存为图片文件
    # ============================================================== #
    print("\n📈 正在生成图表...")

    try:
        from quant_trading.visualization.charts import ChartGenerator

        # 先用策略生成带信号标注的数据
        # generate_signals() 会在原始数据上添加:
        #   - signal 列 ("buy"/"sell"/"hold")
        #   - 指标列 (如 sma_5, sma_20)
        signal_data = strategy.generate_signals(data.copy())

        # 统计信号
        buy_count = (signal_data['signal'] == 'buy').sum()
        sell_count = (signal_data['signal'] == 'sell').sum()
        print(f"   信号统计: 买入 {buy_count} 次, 卖出 {sell_count} 次")

        # ----- 图表1: 交易信号图 -----
        # 显示收盘价折线、均线和买卖信号标注
        print("   [1/4] 生成交易信号图...")
        ChartGenerator.plot_signals(
            signal_data,
            title=f"{symbol} 均线交叉策略信号图",
            # save_path="output/signals.png",  # 取消注释可保存为文件
        )

        # ----- 图表2: K线图 -----
        # 阳线(收盘>开盘)红色, 阴线绿色, 下方成交量柱
        print("   [2/4] 生成K线图...")
        ChartGenerator.plot_candlestick(
            data,
            title=f"{symbol} K线图 ({start_date} ~ {end_date})",
            # save_path="output/candlestick.png",
        )

        # ----- 图表3: 净值曲线 + 回撤 -----
        if not result.equity_curve.empty:
            print("   [3/4] 生成净值曲线图...")
            ChartGenerator.plot_equity_curve(
                result.equity_curve,
                title=f"{symbol} 策略净值曲线",
                # save_path="output/equity.png",
            )

            # ----- 图表4: 月度收益热力图 -----
            print("   [4/4] 生成月度收益热力图...")
            ChartGenerator.plot_monthly_returns(
                result.equity_curve,
                title=f"{symbol} 月度收益热力图",
                # save_path="output/monthly.png",
            )

        print("\n✅ 所有图表生成完成!")

    except ImportError as e:
        print(f"\n⚠️  图表生成跳过 (缺少依赖): {e}")
        print("   请安装: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️  图表生成失败: {e}")

    # ============================================================== #
    #  完成!
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  快速入门示例运行完成!")
    print("")
    print("  接下来你可以:")
    print("    • 修改 symbol 变量试试其他股票 (如 600519 贵州茅台)")
    print("    • 尝试不同策略: rsi, macd, bollinger")
    print("    • 查看 examples/custom_strategy.py 学习自定义策略")
    print("    • 使用命令行工具: quant --help")
    print("    • 组合多策略: composite (多数投票/全票通过/任一)")
    print("=" * 60)


if __name__ == "__main__":
    main()
