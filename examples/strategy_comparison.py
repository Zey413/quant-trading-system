"""
策略对比示例 - 展示如何对比多个策略表现
=============================================

本示例演示如何使用 StrategyComparator 在相同数据上对比多个策略:
1. 加载配置和获取行情数据
2. 使用 StrategyComparator 添加多个策略
3. 运行策略对比
4. 展示排名和格式化摘要
5. 绘制多条净值曲线对比图

运行方式:
    cd <项目根目录>
    pip install -e ".[dev]"
    python examples/strategy_comparison.py

支持的内置策略:
    - ma_crossover: 均线交叉策略
    - rsi: RSI超买超卖策略
    - macd: MACD策略
    - bollinger: 布林带策略
"""

import sys
from pathlib import Path

# ================================================================== #
#  确保项目路径在 sys.path 中 (方便直接运行脚本)
# ================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    print("=" * 60)
    print("  策略对比示例")
    print("=" * 60)

    # ============================================================== #
    #  第1步: 加载配置
    # ============================================================== #
    from quant_trading.core.config import AppConfig

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = AppConfig.from_yaml(str(config_path))

    print(f"\n  初始资金:   {config.backtest.initial_capital:,.0f} 元")
    print(f"  回测区间:   {config.backtest.start_date} ~ {config.backtest.end_date}")

    # ============================================================== #
    #  第2步: 获取行情数据
    # ============================================================== #
    from quant_trading.data.manager import DataManager

    dm = DataManager(config)
    symbol = "000001"
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    print(f"\n  正在获取 {symbol} 行情数据 ({start_date} ~ {end_date})...")

    try:
        data = dm.fetch_daily(symbol, start_date, end_date)
    except Exception as e:
        print(f"\n  数据获取失败: {e}")
        print("  请检查网络连接，或确保已安装 akshare:")
        print("  pip install akshare")
        return

    if data.empty:
        print("  未获取到数据，请检查股票代码和日期范围")
        return

    print(f"  获取到 {len(data)} 条日线数据")

    # ============================================================== #
    #  第3步: 添加多个策略
    #
    #  StrategyComparator 会在相同数据上逐个运行各策略，
    #  每次运行使用独立的 BacktestEngine 实例，确保状态隔离。
    # ============================================================== #

    # 导入策略模块以触发自动注册
    import quant_trading.strategy  # noqa: F401
    from quant_trading.backtest.comparator import StrategyComparator

    comparator = StrategyComparator(config)

    # 添加不同策略（参数可自定义）
    comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
    comparator.add_strategy("ma_crossover", short_window=10, long_window=30)
    comparator.add_strategy("rsi", period=14)
    comparator.add_strategy("macd")
    comparator.add_strategy("bollinger", window=20, num_std=2.0)

    print(f"\n  已添加 5 个策略进行对比")

    # ============================================================== #
    #  第4步: 运行对比
    # ============================================================== #
    print("\n  正在运行策略对比...")
    result = comparator.compare(data, symbol=symbol, rank_by="sharpe_ratio")

    # ============================================================== #
    #  第5步: 展示排名
    # ============================================================== #
    print(result.summary())

    # 也可以直接访问排名 DataFrame 做进一步分析
    print("\n  排名 DataFrame:")
    print(result.ranking.to_string(index=False))

    # 访问某个特定策略的详细结果
    for name, perf in result.results.items():
        print(f"\n  [{name}] 总收益率: {perf.total_return:+.2%}, "
              f"最大回撤: {perf.max_drawdown:.2%}, "
              f"夏普: {perf.sharpe_ratio:.4f}")

    # ============================================================== #
    #  第6步: 绘制净值对比图
    # ============================================================== #
    print("\n  正在生成策略净值对比图...")
    try:
        comparator.plot_comparison(
            result,
            # save_path="output/strategy_comparison.png",  # 取消注释可保存为文件
        )
        print("  图表生成完成!")
    except ImportError as e:
        print(f"  图表生成跳过 (缺少依赖): {e}")
    except Exception as e:
        print(f"  图表生成失败: {e}")

    # ============================================================== #
    #  完成!
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  策略对比示例运行完成!")
    print("")
    print("  你可以:")
    print("    - 添加更多策略进行对比")
    print("    - 修改 rank_by 参数 (如 'total_return', 'calmar_ratio')")
    print("    - 对不同股票或不同时间段运行对比")
    print("    - 使用 result.ranking DataFrame 做自定义分析")
    print("=" * 60)


if __name__ == "__main__":
    main()
