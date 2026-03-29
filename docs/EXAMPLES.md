# 使用示例集

本文档包含 10 个完整的量化交易系统使用示例，涵盖从基础到高级的各种场景。

---

## 目录

- [示例 1：快速上手 — 第一次回测](#示例-1快速上手--第一次回测)
- [示例 2：多策略对比分析](#示例-2多策略对比分析)
- [示例 3：参数优化寻找最佳参数](#示例-3参数优化寻找最佳参数)
- [示例 4：滚动前进分析防止过拟合](#示例-4滚动前进分析防止过拟合)
- [示例 5：多股票组合回测](#示例-5多股票组合回测)
- [示例 6：自定义策略开发](#示例-6自定义策略开发)
- [示例 7：信号质量分析](#示例-7信号质量分析)
- [示例 8：机器学习策略](#示例-8机器学习策略)
- [示例 9：完整回测报告生成](#示例-9完整回测报告生成)
- [示例 10：技术指标分析与可视化](#示例-10技术指标分析与可视化)

---

## 示例 1：快速上手 — 第一次回测

最简单的完整回测流程：获取数据 → 选择策略 → 运行回测 → 查看结果。

### CLI 方式

```bash
# 第一步：获取平安银行 2024 年数据
quant fetch -s 000001 --start 2024-01-01 --end 2024-12-31

# 第二步：使用均线交叉策略回测
quant backtest -st ma_crossover -s 000001 --start 2024-01-01 --end 2024-12-31

# 第三步：生成图表
quant plot -s 000001 -st ma_crossover -t all --save my_first_backtest.png
```

### Python 方式

```python
"""示例1：快速上手 — 第一次回测"""

from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.strategy import StrategyRegistry

# 1. 加载配置
config = AppConfig.from_yaml("config/default.yaml")

# 2. 获取数据
dm = DataManager(config)
data = dm.fetch_daily("000001", "2024-01-01", "2024-12-31")
print(f"获取到 {len(data)} 条行情数据")

# 3. 选择策略（均线交叉，短期5日、长期20日）
strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)

# 4. 运行回测
engine = BacktestEngine(config)
result = engine.run(strategy, data, symbol="000001")

# 5. 查看结果
print("\n========== 回测结果 ==========")
print(f"总收益率:     {result.total_return:.2%}")
print(f"年化收益率:   {result.annualized_return:.2%}")
print(f"夏普比率:     {result.sharpe_ratio:.4f}")
print(f"最大回撤:     {result.max_drawdown:.2%}")
print(f"胜率:         {result.win_rate:.2%}")
print(f"总交易次数:   {result.total_trades}")

# 6. 生成可视化图表
from quant_trading.visualization.charts import ChartGenerator

signal_data = strategy.generate_signals(data.copy())
ChartGenerator.plot_backtest_report(
    data=signal_data,
    equity_curve=result.equity_curve,
    title="000001 均线交叉策略回测",
    save_path="output/example1_report.png",
)
print("\n图表已保存至 output/example1_report.png")
```

---

## 示例 2：多策略对比分析

在相同数据上对比 5 个策略的表现，找出最优策略。

```python
"""示例2：多策略对比分析"""

from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.backtest.comparator import StrategyComparator

# 加载配置 & 获取数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("600519", "2023-01-01", "2024-12-31")

# 创建策略对比器
comparator = StrategyComparator(config)

# 添加多个策略
comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
comparator.add_strategy("rsi", period=14, oversold=30, overbought=70)
comparator.add_strategy("macd", fast=12, slow=26, signal=9)
comparator.add_strategy("bollinger", window=20, num_std=2)
comparator.add_strategy("dual_thrust", lookback=4, k1=0.5, k2=0.5)

# 运行对比（按夏普比率排名）
result = comparator.compare(data, symbol="600519", rank_by="sharpe_ratio")

# 输出对比报告
print(result.summary())

# 绘制净值对比图
comparator.plot_comparison(result, save_path="output/example2_comparison.png")
print("对比图已保存至 output/example2_comparison.png")
```

---

## 示例 3：参数优化寻找最佳参数

使用网格搜索为均线交叉策略找到最优的短期/长期均线窗口。

```python
"""示例3：参数优化"""

from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.backtest.optimizer import GridSearchOptimizer

# 加载配置 & 获取数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("000001", "2023-01-01", "2024-12-31")

# 创建优化器
optimizer = GridSearchOptimizer(config)

# 定义参数搜索空间
param_grid = {
    "short_window": [3, 5, 8, 10, 13],
    "long_window": [15, 20, 25, 30, 40, 60],
}

print(f"总共 {5 * 6} = 30 种参数组合")

# 运行优化（目标：最大化夏普比率）
result = optimizer.optimize(
    strategy_name="ma_crossover",
    param_grid=param_grid,
    data=data,
    symbol="000001",
    metric="sharpe_ratio",
)

# 输出结果
print(f"\n最优参数: {result.best_params}")
print(f"最优夏普比率: {result.best_metric_value:.4f}")

# 查看所有参数组合排名
print("\n========== 参数组合排名 (前10) ==========")
top_results = result.all_results.head(10)
for _, row in top_results.iterrows():
    print(f"  short={int(row.get('short_window', 0)):>3d}, "
          f"long={int(row.get('long_window', 0)):>3d} → "
          f"夏普={row.get('sharpe_ratio', 0):.4f}, "
          f"收益={row.get('total_return', 0):.2%}")

# 使用最优参数重新回测
from quant_trading.strategy import StrategyRegistry
from quant_trading.backtest.engine import BacktestEngine

best_strategy = StrategyRegistry.get("ma_crossover", **result.best_params)
engine = BacktestEngine(config)
best_result = engine.run(best_strategy, data, symbol="000001")
print(f"\n最优参数回测结果:")
print(f"  总收益率: {best_result.total_return:.2%}")
print(f"  最大回撤: {best_result.max_drawdown:.2%}")
```

---

## 示例 4：滚动前进分析防止过拟合

使用 Walk-Forward 分析验证策略参数的稳定性。

```python
"""示例4：滚动前进分析"""

from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.backtest.walk_forward import WalkForwardAnalyzer

# 加载配置 & 获取较长时间段的数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("000001", "2022-01-01", "2024-12-31")
print(f"数据量: {len(data)} 个交易日")

# 创建分析器
analyzer = WalkForwardAnalyzer(config)

# 运行滚动前进分析
result = analyzer.run(
    strategy_name="ma_crossover",
    param_grid={
        "short_window": [3, 5, 10],
        "long_window": [15, 20, 30],
    },
    data=data,
    symbol="000001",
    train_size=120,     # 训练期 120 个交易日（约6个月）
    test_size=60,       # 测试期 60 个交易日（约3个月）
    step_size=60,       # 每次前进 60 个交易日
    optimize_metric="sharpe_ratio",
)

# 输出完整报告
print(result.summary())

# 分析参数稳定性
print("\n========== 参数稳定性分析 ==========")
param_stability = {}
for w in result.windows:
    params_key = str(w.best_params)
    param_stability[params_key] = param_stability.get(params_key, 0) + 1

print("各窗口选出的最优参数分布:")
for params, count in sorted(param_stability.items(), key=lambda x: -x[1]):
    pct = count / len(result.windows) * 100
    print(f"  {params}: {count}次 ({pct:.0f}%)")

# 对比训练期和测试期表现
print("\n========== 训练期 vs 测试期表现 ==========")
for i, w in enumerate(result.windows, 1):
    train_ret = w.train_result.total_return
    test_ret = w.test_result.total_return
    decay = (test_ret - train_ret) / abs(train_ret) * 100 if train_ret != 0 else 0
    print(f"  窗口{i}: 训练={train_ret:.2%}, 测试={test_ret:.2%}, "
          f"衰减={decay:.1f}%")
```

---

## 示例 5：多股票组合回测

同时回测多只股票，构建投资组合并分析相关性。

```python
"""示例5：多股票组合回测"""

from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.backtest.portfolio_backtest import PortfolioBacktester
from quant_trading.strategy import StrategyRegistry

# 加载配置 & 获取多只股票数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)

symbols = {
    "000001": "平安银行",
    "600519": "贵州茅台",
    "000858": "五粮液",
    "601318": "中国平安",
    "000333": "美的集团",
}

data_dict = {}
for code, name in symbols.items():
    try:
        df = dm.fetch_daily(code, "2024-01-01", "2024-12-31")
        if not df.empty:
            data_dict[code] = df
            print(f"  {code} ({name}): {len(df)} 条数据")
    except Exception as e:
        print(f"  {code} ({name}): 获取失败 - {e}")

print(f"\n成功获取 {len(data_dict)} 只股票数据")

# 创建策略
strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)

# 方案1：等权组合
backtester = PortfolioBacktester(config)
equal_result = backtester.run(strategy, data_dict, weights="equal")

print("\n========== 等权组合 ==========")
print(f"组合收益率: {equal_result.overall.total_return:.2%}")
print(f"组合夏普:   {equal_result.overall.sharpe_ratio:.4f}")
print(f"组合回撤:   {equal_result.overall.max_drawdown:.2%}")

# 方案2：自定义权重（重仓消费股）
custom_weights = {code: 0.2 for code in data_dict.keys()}
# 如有茅台和五粮液则加重
if "600519" in custom_weights:
    custom_weights["600519"] = 0.35
if "000858" in custom_weights:
    custom_weights["000858"] = 0.25
# 重新归一化
total_w = sum(custom_weights.values())
custom_weights = {k: v / total_w for k, v in custom_weights.items()}

custom_result = backtester.run(strategy, data_dict, weights=custom_weights)

print("\n========== 自定义权重组合 ==========")
print(f"组合收益率: {custom_result.overall.total_return:.2%}")
print(f"组合夏普:   {custom_result.overall.sharpe_ratio:.4f}")

# 输出完整报告（包含个股绩效和相关性矩阵）
print(equal_result.summary())
```

---

## 示例 6：自定义策略开发

从零开发一个「RSI + 布林带」组合策略。

```python
"""示例6：自定义策略开发"""

from __future__ import annotations
import pandas as pd
from quant_trading.strategy.base import Strategy, StrategyRegistry
from quant_trading.indicators.oscillator import RSI
from quant_trading.indicators.volatility import BollingerBands


@StrategyRegistry.register("rsi_bb")
class RSIBollingerStrategy(Strategy):
    """RSI + 布林带组合策略

    买入条件:
        1. RSI < 30（超卖）
        2. 价格 <= 布林带下轨（超跌）

    卖出条件:
        1. RSI > 70（超买）
        2. 价格 >= 布林带上轨（超涨）
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_window: int = 20,
        bb_std: float = 2.0,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_window = bb_window
        self.bb_std = bb_std

    def get_params(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "bb_window": self.bb_window,
            "bb_std": self.bb_std,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["signal"] = "hold"

        # 数据量检查
        min_periods = max(self.rsi_period, self.bb_window) + 1
        if len(data) < min_periods:
            return data

        # 计算指标
        RSI(period=self.rsi_period).calculate(data)
        BollingerBands(window=self.bb_window, num_std=self.bb_std).calculate(data)

        rsi_col = f"rsi_{self.rsi_period}"

        # 买入：双重确认
        buy_mask = (
            (data[rsi_col] < self.rsi_oversold) &
            (data["close"] <= data["bb_lower"])
        )
        data.loc[buy_mask, "signal"] = "buy"

        # 卖出：双重确认
        sell_mask = (
            (data[rsi_col] > self.rsi_overbought) &
            (data["close"] >= data["bb_upper"])
        )
        data.loc[sell_mask, "signal"] = "sell"

        return data


# ========== 使用自定义策略 ==========

from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.backtest.engine import BacktestEngine

config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("000001", "2024-01-01", "2024-12-31")

# 通过注册表获取
strategy = StrategyRegistry.get("rsi_bb", rsi_period=14, bb_window=20)

# 运行回测
engine = BacktestEngine(config)
result = engine.run(strategy, data, symbol="000001")

print(f"RSI+布林带策略:")
print(f"  总收益率: {result.total_return:.2%}")
print(f"  夏普比率: {result.sharpe_ratio:.4f}")
print(f"  最大回撤: {result.max_drawdown:.2%}")
print(f"  胜率:     {result.win_rate:.2%}")
```

---

## 示例 7：信号质量分析

评估策略信号的预测能力，不需要运行完整回测。

```python
"""示例7：信号质量分析"""

import numpy as np
from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.strategy import StrategyRegistry
from quant_trading.backtest.signal_analyzer import SignalAnalyzer

# 获取数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("000001", "2023-01-01", "2024-12-31")

# 分析不同策略的信号质量
strategies_to_analyze = [
    ("ma_crossover", {"short_window": 5, "long_window": 20}),
    ("rsi", {"period": 14}),
    ("macd", {}),
    ("bollinger", {"window": 20}),
]

analyzer = SignalAnalyzer()

print("========== 策略信号质量对比 ==========\n")

for strategy_name, params in strategies_to_analyze:
    # 生成信号
    strategy = StrategyRegistry.get(strategy_name, **params)
    signal_data = strategy.generate_signals(data.copy())

    # 将字符串信号转为数值
    signal_map = {"buy": 1, "sell": -1, "hold": 0}
    signal_data["signal_num"] = signal_data["signal"].map(signal_map).fillna(0)

    # 准备分析数据
    analysis_data = signal_data[["close", "signal_num"]].copy()
    analysis_data.columns = ["close", "signal"]

    # 分析
    try:
        result = analyzer.analyze(analysis_data, forward_periods=[1, 5, 10])
        print(f"策略: {strategy_name}")
        print(f"  买入信号数: {result.buy_signals}")
        print(f"  卖出信号数: {result.sell_signals}")
        print(f"  买入后平均收益: {result.avg_return_after_buy:.4%}")
        print(f"  买入信号正确率: {result.buy_win_rate:.2%}")
        print(f"  平均持仓天数: {result.avg_holding_period:.1f}")
        print(f"  最大连胜: {result.consecutive_wins}")
        print()
    except Exception as e:
        print(f"策略: {strategy_name} - 分析失败: {e}\n")

# 绘制最佳策略的信号分布图
best_strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)
best_signal_data = best_strategy.generate_signals(data.copy())
best_signal_data["signal_num"] = best_signal_data["signal"].map(signal_map).fillna(0)
analysis_df = best_signal_data[["close", "signal_num"]].copy()
analysis_df.columns = ["close", "signal"]

analyzer.plot_signal_distribution(analysis_df, save_path="output/example7_signals.png")
print("信号分布图已保存至 output/example7_signals.png")
```

---

## 示例 8：机器学习策略

使用随机森林模型构建量化交易策略。

```python
"""示例8：机器学习策略"""

import numpy as np
import pandas as pd
from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.ml import (
    FeatureEngineer,
    RandomForestModel,
    MLPipeline,
    StandardScaler,
    ModelEvaluator,
)
from quant_trading.indicators.utils import add_all_indicators

# 获取数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("000001", "2022-01-01", "2024-12-31")
print(f"数据量: {len(data)} 条")

# 1. 特征工程
fe = FeatureEngineer()
features_df = fe.create_features(data)

# 2. 构造标签：未来5日收益率 > 0 → 买入(1)，否则卖出(0)
future_return = data["close"].pct_change(5).shift(-5)
labels = (future_return > 0).astype(int).values

# 3. 准备特征矩阵
feature_cols = [c for c in features_df.columns
                if c not in ["date", "open", "high", "low", "close", "volume"]]
X = features_df[feature_cols].values

print(f"特征数量: {len(feature_cols)}")

# 4. 清理NaN
valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(labels)
X_valid = X[valid_mask]
y_valid = labels[valid_mask]
print(f"有效样本数: {len(X_valid)}")

# 5. 训练/测试划分（时序划分，不打乱）
split_ratio = 0.7
split_idx = int(len(X_valid) * split_ratio)
X_train, X_test = X_valid[:split_idx], X_valid[split_idx:]
y_train, y_test = y_valid[:split_idx], y_valid[split_idx:]

print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 6. 构建流水线
pipeline = MLPipeline(
    model=RandomForestModel(n_trees=50, max_depth=8),
    scaler=StandardScaler(),
)

# 7. 训练
pipeline.fit(X_train, y_train)
print("模型训练完成")

# 8. 预测 & 评估
train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

evaluator = ModelEvaluator()

print("\n========== 训练集指标 ==========")
train_metrics = evaluator.evaluate(y_train, train_pred)
for name, value in train_metrics.items():
    print(f"  {name}: {value:.4f}")

print("\n========== 测试集指标 ==========")
test_metrics = evaluator.evaluate(y_test, test_pred)
for name, value in test_metrics.items():
    print(f"  {name}: {value:.4f}")

# 9. 检查过拟合
train_acc = train_metrics.get("accuracy", 0)
test_acc = test_metrics.get("accuracy", 0)
overfit_gap = train_acc - test_acc
print(f"\n过拟合差距: {overfit_gap:.4f}")
if overfit_gap > 0.1:
    print("  注意：存在明显过拟合，建议减少模型复杂度")
else:
    print("  过拟合程度可接受")
```

---

## 示例 9：完整回测报告生成

生成包含文本报告、交易记录CSV和净值曲线的完整回测报告。

### CLI 方式

```bash
# 文本报告
quant report -st ma_crossover -s 000001 --start 2024-01-01 --end 2024-12-31

# CSV格式（交易记录 + 净值曲线 + 文本报告）
quant report -st ma_crossover -s 000001 -f csv -o ./reports

# 同时保存图表
quant backtest -st ma_crossover -s 000001 --save reports/chart.png
```

### Python 方式

```python
"""示例9：完整回测报告生成"""

import os
from datetime import datetime
from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.data.export import DataExporter
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.report import ReportGenerator
from quant_trading.strategy import StrategyRegistry
from quant_trading.visualization.charts import ChartGenerator

# 配置
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("000001", "2024-01-01", "2024-12-31")

# 回测
strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)
engine = BacktestEngine(config)
result = engine.run(strategy, data, symbol="000001")

# 创建输出目录
output_dir = "reports"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
prefix = f"{output_dir}/000001_ma_crossover_{timestamp}"

# 1. 生成文本报告
text_report = ReportGenerator.generate_text_report(
    result=result,
    strategy_name="ma_crossover",
    symbol="000001",
    config=config,
    trades=engine.trades,
)
report_path = f"{prefix}_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(text_report)
print(f"文本报告: {report_path}")

# 2. 导出交易记录
trades_path = f"{prefix}_trades.csv"
ReportGenerator.generate_csv_trades(engine.trades, trades_path)
print(f"交易记录: {trades_path}")

# 3. 导出净值曲线
equity_path = f"{prefix}_equity.csv"
ReportGenerator.generate_equity_csv(result.equity_curve, equity_path)
print(f"净值曲线: {equity_path}")

# 4. 导出行情数据（多格式）
DataExporter.to_csv(data, f"{prefix}_data.csv")
DataExporter.to_json(data, f"{prefix}_data.json")
print(f"行情数据: CSV + JSON")

# 5. 生成可视化图表
signal_data = strategy.generate_signals(data.copy())
ChartGenerator.plot_backtest_report(
    data=signal_data,
    equity_curve=result.equity_curve,
    title="000001 均线交叉策略 完整回测报告",
    save_path=f"{prefix}_chart.png",
)
print(f"图表: {prefix}_chart.png")

# 6. 打印报告摘要
print("\n" + text_report)
```

---

## 示例 10：技术指标分析与可视化

一键添加所有指标，分析指标间相关性。

```python
"""示例10：技术指标分析与可视化"""

from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.indicators.utils import (
    add_all_indicators,
    calculate_indicator_correlation,
)
from quant_trading.indicators.trend import SMA, EMA, MACD
from quant_trading.indicators.oscillator import RSI
from quant_trading.indicators.volatility import BollingerBands, ATR
from quant_trading.indicators.momentum import ROC, WilliamsR, CCI

# 获取数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
data = dm.fetch_daily("600519", "2024-01-01", "2024-12-31")
print(f"获取到 {len(data)} 条贵州茅台数据")

# 方式1：逐个添加指标
data_individual = data.copy()
SMA(window=5).calculate(data_individual)
SMA(window=20).calculate(data_individual)
SMA(window=60).calculate(data_individual)
EMA(window=12).calculate(data_individual)
EMA(window=26).calculate(data_individual)
RSI(period=14).calculate(data_individual)
MACD().calculate(data_individual)
BollingerBands(window=20).calculate(data_individual)
ATR(period=14).calculate(data_individual)
ROC(period=12).calculate(data_individual)
WilliamsR(period=14).calculate(data_individual)
CCI(period=20).calculate(data_individual)

print(f"\n逐个添加后的列数: {len(data_individual.columns)}")
print(f"新增指标列: {[c for c in data_individual.columns if c not in data.columns]}")

# 方式2：一键添加所有指标
data_all = data.copy()
add_all_indicators(data_all)
print(f"\n一键添加后的列数: {len(data_all.columns)}")

# 方式3：自定义配置
data_custom = data.copy()
add_all_indicators(data_custom, config={
    "sma": [5, 10, 20],
    "ema": [12, 26],
    "rsi": [14],
    "macd": True,
    "bollinger": [20],
    "atr": [14],
    "obv": False,      # 不添加 OBV
    "vwap": False,      # 不添加 VWAP
    "roc": [12],
    "williams_r": [14],
    "cci": [20],
})
print(f"自定义配置后的列数: {len(data_custom.columns)}")

# 指标相关性分析
indicator_cols = ["rsi_14", "roc_12", "williams_r_14", "cci_20", "macd"]
corr = calculate_indicator_correlation(data_all, indicator_cols)

print("\n========== 指标相关性矩阵 ==========")
print(corr.round(3).to_string())

# 找出高度相关的指标对
print("\n========== 高度相关的指标对 (|r| > 0.7) ==========")
for i in range(len(indicator_cols)):
    for j in range(i + 1, len(indicator_cols)):
        r = corr.iloc[i, j]
        if abs(r) > 0.7:
            print(f"  {indicator_cols[i]} <-> {indicator_cols[j]}: {r:.3f}")

# 查看最新指标值
print("\n========== 最新指标值 ==========")
latest = data_all.iloc[-1]
for col in indicator_cols:
    print(f"  {col}: {latest[col]:.4f}")

# 绘制K线图
from quant_trading.visualization.charts import ChartGenerator
ChartGenerator.plot_candlestick(
    data_all,
    title="600519 贵州茅台 K线图（含指标）",
    save_path="output/example10_kline.png",
)
print("\nK线图已保存至 output/example10_kline.png")
```
