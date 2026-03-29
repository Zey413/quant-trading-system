"""
风控管理示例 - 展示止损止盈和仓位管理
=========================================

本示例演示量化交易系统的风控管理功能:
1. 配置不同的风控参数 (止损/止盈/仓位比例)
2. 对比有无风控的回测结果
3. 展示止损/止盈如何影响交易表现

运行方式:
    cd <项目根目录>
    pip install -e ".[dev]"
    python examples/risk_management.py

核心概念:
    RiskManager 是统一的风控入口:
    - 信号验证: 检查是否超过持仓限制
    - 仓位管理: 根据 fixed_ratio / kelly / atr 计算下单数量
    - 止损止盈: 每个交易日检查持仓是否触发止损/止盈阈值
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
    print("  风控管理示例 - 止损止盈与仓位管理")
    print("=" * 60)

    # ============================================================== #
    #  第1步: 获取数据
    # ============================================================== #
    from quant_trading.core.config import (
        AppConfig,
        BacktestConfig,
        BrokerConfig,
        RiskConfig,
    )
    from quant_trading.data.manager import DataManager

    # 使用默认配置加载数据
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    base_config = AppConfig.from_yaml(str(config_path))

    dm = DataManager(base_config)
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

    # 导入策略模块以触发注册
    import quant_trading.strategy  # noqa: F401
    from quant_trading.backtest.engine import BacktestEngine
    from quant_trading.strategy import StrategyRegistry

    strategy_name = "ma_crossover"
    strategy_params = {"short_window": 5, "long_window": 20}

    # ============================================================== #
    #  第2步: 配置不同的风控参数
    #
    #  我们创建三种配置进行对比:
    #  A) 无风控 (宽松配置: 大止损、大仓位)
    #  B) 标准风控 (5%止损, 15%止盈, 10%仓位)
    #  C) 激进风控 (3%止损, 10%止盈, 20%仓位)
    # ============================================================== #

    # 共享的券商和回测配置
    broker_config = BrokerConfig(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax_rate=0.001,
        slippage=0.001,
        lot_size=100,
        t_plus_1=True,
    )
    backtest_config = BacktestConfig(
        initial_capital=1_000_000.0,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=0.025,
    )

    # 配置A: 无风控（极宽松参数模拟无风控效果）
    config_no_risk = AppConfig(
        broker=broker_config,
        backtest=backtest_config,
        risk=RiskConfig(
            max_position_pct=0.95,           # 几乎不限单只持仓
            max_total_position_pct=0.95,     # 几乎不限总持仓
            stop_loss_pct=0.99,              # 极大止损 (等于不止损)
            take_profit_pct=0.99,            # 极大止盈 (等于不止盈)
            position_sizing_method="fixed_ratio",
            fixed_ratio=0.3,                 # 30%仓位
        ),
    )

    # 配置B: 标准风控
    config_standard = AppConfig(
        broker=broker_config,
        backtest=backtest_config,
        risk=RiskConfig(
            max_position_pct=0.3,
            max_total_position_pct=0.8,
            stop_loss_pct=0.05,              # 5%止损
            take_profit_pct=0.15,            # 15%止盈
            position_sizing_method="fixed_ratio",
            fixed_ratio=0.1,                 # 10%仓位
        ),
    )

    # 配置C: 激进风控
    config_tight = AppConfig(
        broker=broker_config,
        backtest=backtest_config,
        risk=RiskConfig(
            max_position_pct=0.3,
            max_total_position_pct=0.8,
            stop_loss_pct=0.03,              # 3%止损 (更紧)
            take_profit_pct=0.10,            # 10%止盈 (更早锁利)
            position_sizing_method="fixed_ratio",
            fixed_ratio=0.2,                 # 20%仓位
        ),
    )

    configurations = [
        ("无风控", config_no_risk),
        ("标准风控 (5%止损/15%止盈)", config_standard),
        ("激进风控 (3%止损/10%止盈)", config_tight),
    ]

    print(f"\n  风控配置:")
    print(f"  {'配置名称':<32} {'止损':>6} {'止盈':>6} {'仓位比例':>8}")
    print(f"  {'-' * 32} {'-' * 6} {'-' * 6} {'-' * 8}")
    for name, cfg in configurations:
        print(
            f"  {name:<32} "
            f"{cfg.risk.stop_loss_pct:>6.0%} "
            f"{cfg.risk.take_profit_pct:>6.0%} "
            f"{cfg.risk.fixed_ratio:>8.0%}"
        )

    # ============================================================== #
    #  第3步: 对比有无风控的回测结果
    # ============================================================== #
    print(f"\n  正在运行策略 '{strategy_name}' 对比测试...")

    results = {}
    for name, cfg in configurations:
        strategy = StrategyRegistry.get(strategy_name, **strategy_params)
        engine = BacktestEngine(cfg)
        result = engine.run(strategy, data.copy(), symbol=symbol)
        results[name] = result
        print(f"    [{name}] 完成")

    # ============================================================== #
    #  第4步: 展示对比结果
    # ============================================================== #
    print("\n" + "=" * 80)
    print("                    风控对比结果")
    print("=" * 80)

    col_width = 24
    names = list(results.keys())

    # 表头
    header = f"{'指标':<16}"
    for name in names:
        # 截断长名称
        short_name = name[:col_width - 2]
        header += f"  {short_name:>{col_width - 2}}"
    print(header)
    print(f"{'-' * 16}" + (f"  {'-' * (col_width - 2)}") * len(names))

    metrics_to_show = [
        ("total_return", "总收益率", True),
        ("annualized_return", "年化收益率", True),
        ("max_drawdown", "最大回撤", True),
        ("max_drawdown_duration", "最大回撤天数", False),
        ("volatility", "年化波动率", True),
        ("sharpe_ratio", "夏普比率", False),
        ("sortino_ratio", "Sortino比率", False),
        ("calmar_ratio", "Calmar比率", False),
        ("total_trades", "交易次数", False),
        ("win_rate", "胜率", True),
        ("profit_loss_ratio", "盈亏比", False),
    ]

    for key, label, is_pct in metrics_to_show:
        row = f"{label:<16}"
        for name in names:
            value = getattr(results[name], key, 0.0)
            if is_pct:
                cell = f"{value:>{col_width - 2}.2%}"
            elif key in ("total_trades", "max_drawdown_duration"):
                cell = f"{int(value):>{col_width - 2}d}"
            else:
                cell = f"{value:>{col_width - 2}.4f}"
            row += f"  {cell}"
        print(row)

    print("=" * 80)

    # ============================================================== #
    #  第5步: 分析风控效果
    # ============================================================== #
    print("\n  风控效果分析:")

    no_risk = results["无风控"]
    standard = results["标准风控 (5%止损/15%止盈)"]
    tight = results["激进风控 (3%止损/10%止盈)"]

    # 最大回撤改善
    if no_risk.max_drawdown > 0:
        std_dd_improve = (no_risk.max_drawdown - standard.max_drawdown) / no_risk.max_drawdown
        tight_dd_improve = (no_risk.max_drawdown - tight.max_drawdown) / no_risk.max_drawdown
        print(f"    标准风控 vs 无风控:")
        print(f"      最大回撤变化: {std_dd_improve:+.1%}")
        print(f"      夏普比率变化: {standard.sharpe_ratio - no_risk.sharpe_ratio:+.4f}")
        print(f"    激进风控 vs 无风控:")
        print(f"      最大回撤变化: {tight_dd_improve:+.1%}")
        print(f"      夏普比率变化: {tight.sharpe_ratio - no_risk.sharpe_ratio:+.4f}")

    print(f"\n    交易频率对比:")
    print(f"      无风控:   {no_risk.total_trades} 笔")
    print(f"      标准风控: {standard.total_trades} 笔")
    print(f"      激进风控: {tight.total_trades} 笔")
    print(f"    (止损/止盈可能触发额外的卖出交易)")

    # ============================================================== #
    #  第6步: 可视化对比
    # ============================================================== #
    print("\n  正在生成净值对比图...")

    try:
        from quant_trading.backtest.comparator import ComparisonResult, StrategyComparator

        comparison_result = ComparisonResult(
            results=results,
            ranking=None,
            rank_metric="sharpe_ratio",
        )
        # 手动构建 ranking DataFrame
        import pandas as pd
        ranking_rows = []
        for name, r in results.items():
            ranking_rows.append({
                "strategy": name,
                "sharpe_ratio": r.sharpe_ratio,
                "total_return": r.total_return,
                "max_drawdown": r.max_drawdown,
            })
        comparison_result.ranking = pd.DataFrame(ranking_rows).sort_values(
            "sharpe_ratio", ascending=False
        ).reset_index(drop=True)

        # 使用 StrategyComparator 的 plot 方法
        dummy_comparator = StrategyComparator(base_config)
        dummy_comparator.plot_comparison(
            comparison_result,
            # save_path="output/risk_comparison.png",  # 取消注释可保存为文件
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
    print("  风控管理示例运行完成!")
    print("")
    print("  关键结论:")
    print("    - 止损可以控制最大回撤，但可能减少收益")
    print("    - 止盈可以锁定利润，但可能错过大行情")
    print("    - 仓位大小直接影响收益和风险的放大倍数")
    print("    - 最优的风控参数需要结合市场环境和策略特点")
    print("")
    print("  进阶练习:")
    print("    - 尝试不同的止损比例 (2%~10%)")
    print("    - 使用 Kelly 公式或 ATR 动态仓位管理")
    print("    - 在趋势市和震荡市分别测试风控效果")
    print("=" * 60)


if __name__ == "__main__":
    main()
