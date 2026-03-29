"""
参数优化示例 - 展示如何使用 GridSearchOptimizer
=================================================

本示例演示参数优化的完整流程:
1. 定义参数网格 (参数搜索空间)
2. 运行网格搜索优化
3. 展示最优参数和所有结果
4. 用最优参数运行完整回测并查看绩效

运行方式:
    cd <项目根目录>
    pip install -e ".[dev]"
    python examples/parameter_optimization.py

核心概念:
    GridSearchOptimizer 对策略参数进行穷举搜索，
    为每组参数运行独立的完整回测，
    并按指定指标（如夏普比率）选出最优参数组合。
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
    print("  参数优化示例 - GridSearchOptimizer")
    print("=" * 60)

    # ============================================================== #
    #  第1步: 加载配置并获取数据
    # ============================================================== #
    from quant_trading.core.config import AppConfig
    from quant_trading.data.manager import DataManager

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = AppConfig.from_yaml(str(config_path))

    dm = DataManager(config)
    symbol = "000001"
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    print(f"\n  正在获取 {symbol} 行情数据 ({start_date} ~ {end_date})...")

    try:
        data = dm.fetch_daily(symbol, start_date, end_date)
    except Exception as e:
        print(f"\n  数据获取失败: {e}")
        print("  请确保已安装 akshare: pip install akshare")
        return

    if data.empty:
        print("  未获取到数据")
        return

    print(f"  获取到 {len(data)} 条日线数据")

    # ============================================================== #
    #  第2步: 定义参数网格
    #
    #  参数网格是一个字典，键是策略参数名，值是参数候选值列表。
    #  优化器会遍历所有组合（笛卡尔积）。
    #
    #  注意:
    #  - 参数组合数 = 各参数候选值个数的乘积
    #  - 组合数过多会导致运行时间过长
    #  - 不合法的组合（如 short >= long）会被自动跳过
    # ============================================================== #

    # 导入策略模块以触发自动注册
    import quant_trading.strategy  # noqa: F401
    from quant_trading.backtest.optimizer import GridSearchOptimizer

    # 均线交叉策略的参数网格
    param_grid = {
        "short_window": [3, 5, 8, 10],    # 短期均线周期
        "long_window": [15, 20, 25, 30],   # 长期均线周期
    }

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"\n  参数网格:")
    for name, values in param_grid.items():
        print(f"    {name}: {values}")
    print(f"  总组合数: {total_combinations}")
    print(f"  (不合法组合如 short >= long 将被自动跳过)")

    # ============================================================== #
    #  第3步: 运行网格搜索
    #
    #  GridSearchOptimizer 支持以下优化指标:
    #  - sharpe_ratio (夏普比率，默认)
    #  - total_return (总收益率)
    #  - annualized_return (年化收益率)
    #  - sortino_ratio (Sortino比率)
    #  - calmar_ratio (Calmar比率)
    #  - win_rate (胜率)
    #  - max_drawdown (最大回撤，越小越好)
    # ============================================================== #
    optimizer = GridSearchOptimizer(config)

    print(f"\n  正在运行网格搜索 (优化指标: sharpe_ratio)...")
    result = optimizer.optimize(
        strategy_name="ma_crossover",
        param_grid=param_grid,
        data=data,
        symbol=symbol,
        metric="sharpe_ratio",
    )

    # ============================================================== #
    #  第4步: 展示最优参数
    # ============================================================== #
    print(result.summary())

    # 查看所有结果排名
    print("\n  所有参数组合结果 (按夏普比率降序):")
    print(f"  {'短周期':>8} {'长周期':>8} {'夏普比率':>10} {'总收益率':>10} {'最大回撤':>10}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")

    for _, row in result.all_results.iterrows():
        print(
            f"  {int(row['short_window']):>8} "
            f"{int(row['long_window']):>8} "
            f"{row['sharpe_ratio']:>10.4f} "
            f"{row['total_return']:>10.2%} "
            f"{row['max_drawdown']:>10.2%}"
        )

    # ============================================================== #
    #  第5步: 用最优参数运行完整回测
    # ============================================================== #
    from quant_trading.backtest.engine import BacktestEngine
    from quant_trading.strategy import StrategyRegistry

    best_params = result.best_params
    print(f"\n  使用最优参数运行完整回测...")
    print(f"  最优参数: {best_params}")

    best_strategy = StrategyRegistry.get("ma_crossover", **best_params)
    engine = BacktestEngine(config)
    best_result = engine.run(best_strategy, data, symbol=symbol)

    print(best_result.summary())

    # ============================================================== #
    #  额外: 用 total_return 优化
    # ============================================================== #
    print("\n" + "-" * 60)
    print("  额外: 以总收益率为目标重新优化")
    print("-" * 60)

    result_tr = optimizer.optimize(
        strategy_name="ma_crossover",
        param_grid=param_grid,
        data=data,
        symbol=symbol,
        metric="total_return",
    )

    print(f"\n  最优参数 (总收益率): {result_tr.best_params}")
    print(f"  最优总收益率: {result_tr.best_metric_value:.2%}")

    # 比较两种优化目标的结果差异
    if result.best_params != result_tr.best_params:
        print(f"\n  不同优化目标可能产生不同的最优参数!")
        print(f"    夏普最优参数:   {result.best_params}")
        print(f"    总收益最优参数: {result_tr.best_params}")
    else:
        print(f"\n  两种优化目标产生了相同的最优参数: {result.best_params}")

    # ============================================================== #
    #  完成!
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  参数优化示例运行完成!")
    print("")
    print("  进阶练习:")
    print("    - 尝试优化 RSI 策略的 period 和 oversold/overbought 参数")
    print("    - 用 calmar_ratio 作为优化目标")
    print("    - 在不同时间段上验证最优参数的稳定性 (走样本外测试)")
    print("    - 增加参数粒度 (更多候选值) 以找到更精确的最优点")
    print("=" * 60)


if __name__ == "__main__":
    main()
