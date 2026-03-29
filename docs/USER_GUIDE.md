# 用户手册

本手册面向量化交易系统的使用者，提供从安装到实战的完整指导。

---

## 目录

- [系统要求](#系统要求)
- [安装配置](#安装配置)
- [数据获取](#数据获取)
- [运行回测](#运行回测)
- [策略选择指南](#策略选择指南)
- [参数优化](#参数优化)
- [滚动前进分析](#滚动前进分析)
- [多股票组合回测](#多股票组合回测)
- [信号质量分析](#信号质量分析)
- [策略对比](#策略对比)
- [机器学习策略](#机器学习策略)
- [数据导出](#数据导出)
- [可视化图表](#可视化图表)
- [配置详解](#配置详解)
- [FAQ](#faq)

---

## 系统要求

| 项目 | 要求 |
|:-----|:-----|
| 操作系统 | Windows / macOS / Linux |
| Python | 3.10 或更高版本 |
| 内存 | 建议 4GB 以上 |
| 磁盘 | 建议 1GB 以上（含数据缓存） |
| 网络 | 首次获取数据需要网络连接 |

### 依赖库

核心依赖会在安装时自动安装：

| 库 | 版本 | 用途 |
|:---|:-----|:-----|
| pandas | >= 2.0.0 | 数据处理 |
| numpy | >= 1.24.0 | 数值计算 |
| matplotlib | >= 3.7.0 | 图表绘制 |
| pydantic | >= 2.0.0 | 配置验证 |
| pyyaml | >= 6.0 | YAML配置解析 |
| click | >= 8.1.0 | 命令行框架 |
| rich | >= 13.0.0 | 终端美化 |
| pyarrow | >= 12.0.0 | Parquet缓存 |
| akshare | >= 1.10.0 | 免费A股数据 |
| tushare | >= 1.4.0 | 专业A股数据 |

---

## 安装配置

### 方式一：pip 安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/zey413/quant-trading-system.git
cd quant-trading-system

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 3. 安装项目
pip install -e ".[dev]"

# 4. 验证
quant --help
```

### 方式二：Docker 安装

```bash
# 构建镜像
docker build -t quant-trading .

# 运行容器
docker run -it quant-trading quant --help
```

### 方式三：Docker Compose

```bash
docker-compose up -d
docker-compose exec quant quant --help
```

### 配置数据源

系统默认使用 AKShare（免费），无需任何配置即可使用。

如需使用 Tushare Pro（数据更稳定，需Token）：

1. 注册 [Tushare Pro](https://tushare.pro/) 获取 Token
2. 编辑 `config/default.yaml`：

```yaml
data_source:
  default_source: "tushare"
  tushare_token: "你的Token"
```

或通过命令行指定数据源：

```bash
quant fetch -s 000001 --source tushare
```

### 验证安装

```bash
# 查看帮助
quant --help

# 列出所有策略
quant strategies

# 获取测试数据
quant fetch -s 000001 --start 2024-01-01 --end 2024-06-30

# 运行一次回测
quant backtest -st ma_crossover -s 000001 --start 2024-01-01 --end 2024-06-30
```

---

## 数据获取

### CLI 获取数据

```bash
# 基础用法
quant fetch -s 000001 --start 2024-01-01 --end 2025-01-01

# 指定数据源
quant fetch -s 600519 --source akshare

# 控制显示行数
quant fetch -s 000001 -n 50    # 显示最近50条
quant fetch -s 000001 -n 0     # 显示全部
```

### Python API 获取数据

```python
from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager

# 加载配置
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)

# 获取日线数据
df = dm.fetch_daily("000001", "2024-01-01", "2025-01-01")
print(df.head())
#         date   open   high    low  close    volume
# 0 2024-01-02  10.15  10.28  10.10  10.20  12345678
# 1 2024-01-03  10.22  10.35  10.18  10.30  15678901
# ...

# 切换数据源
dm.set_source("tushare")
df_tushare = dm.fetch_daily("000001", "2024-01-01", "2025-01-01")
```

### 数据缓存

系统默认开启 Parquet 缓存，相同数据第二次获取会直接读取本地文件，速度极快。

缓存目录默认为 `data_cache/`，可在配置文件中修改：

```yaml
data_source:
  cache_dir: "data_cache"
  cache_enabled: true
```

手动清除缓存：

```bash
rm -rf data_cache/
```

### 数据格式标准

所有数据源返回的 DataFrame 都会被标准化为以下格式：

| 列名 | 类型 | 说明 |
|:-----|:-----|:-----|
| `date` | datetime64 | 交易日期 |
| `open` | float64 | 开盘价 |
| `high` | float64 | 最高价 |
| `low` | float64 | 最低价 |
| `close` | float64 | 收盘价 |
| `volume` | int64 | 成交量 |

---

## 运行回测

### CLI 回测

```bash
# 基础回测
quant backtest -st ma_crossover -s 000001

# 自定义日期和资金
quant backtest -st rsi -s 600519 --start 2023-01-01 --end 2024-12-31 --capital 500000

# 保存回测图表
quant backtest -st macd -s 000001 --save report.png
```

### Python API 回测

```python
from quant_trading.core.config import AppConfig
from quant_trading.data.manager import DataManager
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.strategy import StrategyRegistry

# 加载配置
config = AppConfig.from_yaml("config/default.yaml")

# 获取数据
dm = DataManager(config)
data = dm.fetch_daily("000001", "2024-01-01", "2025-01-01")

# 获取策略实例
strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)

# 运行回测
engine = BacktestEngine(config)
result = engine.run(strategy, data, symbol="000001")

# 查看结果
print(f"总收益率:   {result.total_return:.2%}")
print(f"年化收益率: {result.annualized_return:.2%}")
print(f"夏普比率:   {result.sharpe_ratio:.4f}")
print(f"最大回撤:   {result.max_drawdown:.2%}")
print(f"胜率:       {result.win_rate:.2%}")
print(f"交易次数:   {result.total_trades}")
```

### 回测结果解读

回测结果 `PerformanceResult` 包含以下指标：

| 指标 | 含义 | 参考标准 |
|:-----|:-----|:---------|
| `total_return` | 总收益率 | 正值盈利，负值亏损 |
| `annualized_return` | 年化收益率 | > 10% 较好 |
| `max_drawdown` | 最大回撤 | < 20% 较好 |
| `sharpe_ratio` | 夏普比率 | > 1.0 良好，> 2.0 优秀 |
| `sortino_ratio` | Sortino比率 | > 1.5 较好 |
| `calmar_ratio` | Calmar比率 | > 1.0 较好 |
| `win_rate` | 胜率 | > 50% 正胜率 |
| `profit_loss_ratio` | 盈亏比 | > 1.5 较好 |

---

## 策略选择指南

### 策略分类与适用场景

#### 趋势跟踪类

| 策略 | 适用市场 | 优势 | 劣势 |
|:-----|:---------|:-----|:-----|
| **均线交叉** | 趋势明确的市场 | 简单可靠，参数少 | 震荡市信号多，滞后 |
| **MACD** | 中长期趋势 | 过滤假信号较好 | 大牛/大熊市反应慢 |
| **海龟交易** | 强趋势行情 | 经典验证，仓位管理 | 震荡市回撤大 |

#### 动量/突破类

| 策略 | 适用市场 | 优势 | 劣势 |
|:-----|:---------|:-----|:-----|
| **Dual Thrust** | 日内/短期波动 | 捕捉突破行情 | 假突破风险 |
| **RSI** | 超买超卖 | 逆势布局 | 强势单边行情易逆势 |

#### 均值回归类

| 策略 | 适用市场 | 优势 | 劣势 |
|:-----|:---------|:-----|:-----|
| **布林带** | 震荡区间 | 自适应波动 | 趋势市可能连续止损 |
| **均值回归** | 范围波动 | 数学基础扎实 | 趋势变化时表现差 |

#### 组合/高级类

| 策略 | 适用市场 | 优势 | 劣势 |
|:-----|:---------|:-----|:-----|
| **复合策略** | 任何市场 | 降低单一策略风险 | 可能错过强信号 |
| **ML策略** | 复杂模式 | 自动发现规律 | 过拟合风险，需大量数据 |

### 策略推荐

- **新手入门**: 均线交叉 (`ma_crossover`) — 参数少，逻辑清晰
- **稳健交易**: 复合策略 (`composite`) — 多策略投票降低风险
- **追求收益**: 海龟交易 (`turtle`) — 经典趋势策略，仓位管理完善
- **短期交易**: Dual Thrust (`dual_thrust`) — 适合日内/短期波动
- **量化研究**: 机器学习策略 — 数据驱动的信号生成

---

## 参数优化

### CLI 参数优化

```bash
# 优化均线参数
quant optimize -st ma_crossover -s 000001 \
  -p '{"short_window":[3,5,10],"long_window":[15,20,30]}'

# 指定优化目标（默认为夏普比率）
quant optimize -st rsi -s 600519 \
  -p '{"period":[10,14,20]}' -m total_return
```

### Python API 参数优化

```python
from quant_trading.backtest.optimizer import GridSearchOptimizer

optimizer = GridSearchOptimizer(config)

# 定义参数搜索空间
param_grid = {
    "short_window": [3, 5, 8, 10, 13],
    "long_window": [15, 20, 25, 30, 40],
}

# 运行优化
result = optimizer.optimize(
    strategy_name="ma_crossover",
    param_grid=param_grid,
    data=data,
    symbol="000001",
    metric="sharpe_ratio",
)

# 查看结果
print(f"最优参数: {result.best_params}")
print(f"最优夏普: {result.best_metric_value:.4f}")

# 查看所有结果排序
print(result.all_results.to_string())
```

### 可用优化指标

| 指标名 | 说明 | 越大越好 |
|:-------|:-----|:---------|
| `sharpe_ratio` | 夏普比率（默认） | 是 |
| `total_return` | 总收益率 | 是 |
| `annualized_return` | 年化收益率 | 是 |
| `sortino_ratio` | Sortino比率 | 是 |
| `calmar_ratio` | Calmar比率 | 是 |
| `max_drawdown` | 最大回撤 | 否（越小越好） |
| `volatility` | 年化波动率 | 否（越小越好） |
| `win_rate` | 胜率 | 是 |

### 优化注意事项

1. **过拟合风险**: 参数组合过多会导致过拟合，建议使用滚动前进分析验证
2. **数据量要求**: 至少需要 100 个交易日的数据
3. **计算时间**: 参数组合数 = 各参数候选值数量的乘积

---

## 滚动前进分析

滚动前进分析 (Walk-Forward Analysis) 是防止参数优化过拟合的重要工具。

### 原理

1. 将数据按时间分为多个「训练期 + 测试期」窗口
2. 在每个训练期内使用网格搜索找到最优参数
3. 用训练期的最优参数在测试期验证效果
4. 拼接所有测试期净值，计算整体绩效

### 使用方法

```python
from quant_trading.backtest.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(config)
result = analyzer.run(
    strategy_name="ma_crossover",
    param_grid={
        "short_window": [3, 5, 10],
        "long_window": [15, 20, 30],
    },
    data=df,
    symbol="000001",
    train_size=120,     # 训练期 120 个交易日（约6个月）
    test_size=60,       # 测试期 60 个交易日（约3个月）
    step_size=60,       # 每次前进 60 个交易日
    optimize_metric="sharpe_ratio",
)

# 查看摘要
print(result.summary())

# 查看每个窗口的详细结果
for i, w in enumerate(result.windows):
    print(f"窗口{i+1}: 训练 {w.train_start}~{w.train_end}, "
          f"测试收益: {w.test_result.total_return:.2%}, "
          f"最优参数: {w.best_params}")

# 整体测试期绩效
print(f"整体测试收益: {result.overall_performance.total_return:.2%}")
print(f"整体测试夏普: {result.overall_performance.sharpe_ratio:.4f}")
```

### 窗口参数建议

| 参数 | 建议值 | 说明 |
|:-----|:-------|:-----|
| `train_size` | 120-250 | 6-12个月，足够训练 |
| `test_size` | 40-60 | 2-3个月，足够验证 |
| `step_size` | 与test_size相同 | 窗口不重叠 |

---

## 多股票组合回测

### 使用方法

```python
from quant_trading.backtest.portfolio_backtest import PortfolioBacktester
from quant_trading.strategy import StrategyRegistry

# 准备多只股票的数据
data_dict = {
    "000001": dm.fetch_daily("000001", "2024-01-01", "2025-01-01"),
    "600519": dm.fetch_daily("600519", "2024-01-01", "2025-01-01"),
    "000858": dm.fetch_daily("000858", "2024-01-01", "2025-01-01"),
}

# 创建策略
strategy = StrategyRegistry.get("ma_crossover", short_window=5, long_window=20)

# 等权组合回测
backtester = PortfolioBacktester(config)
result = backtester.run(strategy, data_dict, weights="equal")
print(result.summary())

# 自定义权重
result = backtester.run(strategy, data_dict, weights={
    "000001": 0.4,  # 平安银行 40%
    "600519": 0.4,  # 贵州茅台 40%
    "000858": 0.2,  # 五粮液 20%
})
print(result.summary())
```

### 结果解读

`PortfolioBacktestResult` 包含：
- `overall`: 组合整体绩效
- `per_stock`: 每只股票的单独绩效
- `stock_weights`: 各股票权重
- `correlation_matrix`: 股票日收益率相关性矩阵

---

## 信号质量分析

信号分析器可以独立评估策略信号的预测质量，无需完整回测。

```python
from quant_trading.backtest.signal_analyzer import SignalAnalyzer
from quant_trading.strategy import StrategyRegistry

# 生成信号
strategy = StrategyRegistry.get("ma_crossover")
signal_data = strategy.generate_signals(data.copy())

# 将 "buy"/"sell"/"hold" 转换为数字信号
signal_data["signal_num"] = signal_data["signal"].map(
    {"buy": 1, "sell": -1, "hold": 0}
)

# 分析信号质量
analyzer = SignalAnalyzer()
result = analyzer.analyze(
    signal_data.rename(columns={"signal_num": "signal"}),
    forward_periods=[1, 5, 10],
)
print(result.summary())

# 绘制信号分布图
analyzer.plot_signal_distribution(
    signal_data.rename(columns={"signal_num": "signal"}),
    save_path="signal_analysis.png",
)
```

---

## 策略对比

```python
from quant_trading.backtest.comparator import StrategyComparator

comparator = StrategyComparator(config)

# 添加多个策略
comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
comparator.add_strategy("rsi", period=14)
comparator.add_strategy("macd")
comparator.add_strategy("bollinger", window=20, num_std=2)
comparator.add_strategy("dual_thrust", lookback=4)

# 运行对比
result = comparator.compare(data, symbol="000001", rank_by="sharpe_ratio")

# 查看对比报告
print(result.summary())

# 绘制净值对比图
comparator.plot_comparison(result, save_path="comparison.png")
```

---

## 机器学习策略

### 特征工程

```python
from quant_trading.ml import FeatureEngineer

fe = FeatureEngineer()
features_df = fe.create_features(data)
print(f"生成了 {len(features_df.columns)} 个特征")
```

### 模型训练

```python
from quant_trading.ml import (
    RandomForestModel, MLPipeline, StandardScaler, ModelEvaluator
)

# 准备数据
X_train, X_test = features_df[:200], features_df[200:]
y_train, y_test = labels[:200], labels[200:]

# 构建流水线
pipeline = MLPipeline(
    model=RandomForestModel(n_trees=100, max_depth=10),
    scaler=StandardScaler(),
)

# 训练
pipeline.fit(X_train.values, y_train)

# 预测
predictions = pipeline.predict(X_test.values)

# 评估
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_test, predictions)
print(f"准确率: {metrics['accuracy']:.2%}")
```

---

## 数据导出

```python
from quant_trading.data.export import DataExporter

# 导出行情数据
DataExporter.to_csv(df, "output/stock_data.csv")
DataExporter.to_excel(df, "output/stock_data.xlsx")
DataExporter.to_json(df, "output/stock_data.json")

# 导出回测结果
equity_df = result.equity_curve.to_frame("equity")
DataExporter.to_csv(equity_df, "output/equity_curve.csv")
```

---

## 可视化图表

### CLI 生成图表

```bash
# K线图
quant plot -s 000001 -t candlestick --save kline.png

# 交易信号图
quant plot -s 000001 -st ma_crossover -t signals --save signals.png

# 所有图表
quant plot -s 000001 -st rsi -t all --save charts.png
```

### Python API 生成图表

```python
from quant_trading.visualization.charts import ChartGenerator

# K线图
ChartGenerator.plot_candlestick(data, title="000001 K线图", save_path="kline.png")

# 交易信号图
signal_data = strategy.generate_signals(data.copy())
ChartGenerator.plot_signals(signal_data, title="交易信号", save_path="signals.png")

# 净值曲线
ChartGenerator.plot_equity_curve(
    result.equity_curve,
    title="净值曲线",
    save_path="equity.png",
)

# 回撤图
ChartGenerator.plot_drawdown(
    result.drawdown_series,
    title="回撤分析",
    save_path="drawdown.png",
)

# 四合一回测报告
ChartGenerator.plot_backtest_report(
    data=signal_data,
    equity_curve=result.equity_curve,
    title="000001 - ma_crossover 回测报告",
    save_path="full_report.png",
)
```

### 图表类型

| 图表 | 方法 | 说明 |
|:-----|:-----|:-----|
| K线图 | `plot_candlestick()` | 蜡烛图 + 成交量（红涨绿跌） |
| 信号图 | `plot_signals()` | 价格 + 均线 + 买卖标记 |
| 净值曲线 | `plot_equity_curve()` | 净值走势 + 回撤区域填充 |
| 回撤图 | `plot_drawdown()` | 回撤百分比时序 + 最大回撤标注 |
| 月度热力图 | `plot_monthly_returns()` | 年×月收益热力矩阵 |
| 报告面板 | `plot_backtest_report()` | 四合一综合面板 |
| 信号分布 | `SignalAnalyzer.plot_signal_distribution()` | 信号时序 + 收益分布 |

---

## 配置详解

### 完整配置文件

```yaml
# ===== 数据源配置 =====
data_source:
  default_source: "akshare"       # 数据源: akshare(免费) / tushare(需token)
  tushare_token: ""               # Tushare Pro Token
  cache_dir: "data_cache"         # 缓存目录
  cache_enabled: true             # 是否启用缓存

# ===== A股交易规则 =====
broker:
  commission_rate: 0.0003         # 佣金率 万三 (范围: 0 ~ 0.01)
  min_commission: 5.0             # 最低佣金 5元
  stamp_tax_rate: 0.001           # 印花税率 千一(仅卖出)
  slippage: 0.001                 # 滑点 千一
  lot_size: 100                   # 最小交易单位 100股
  trading_days_per_year: 244      # 年交易日数
  t_plus_1: true                  # T+1 交易规则

# ===== 回测参数 =====
backtest:
  initial_capital: 1000000.0      # 初始资金（元）
  start_date: "2024-01-01"        # 起始日期 (YYYY-MM-DD)
  end_date: "2025-12-31"          # 结束日期
  benchmark: "000300"             # 基准指数
  risk_free_rate: 0.025           # 无风险利率

# ===== 风控参数 =====
risk:
  max_position_pct: 0.3           # 单只最大仓位 30%
  max_total_position_pct: 0.8     # 总仓位上限 80%
  stop_loss_pct: 0.05             # 止损线 5%
  take_profit_pct: 0.15           # 止盈线 15%
  position_sizing_method: "fixed_ratio"  # 仓位算法
  fixed_ratio: 0.1                # 固定比例仓位
```

### 仓位管理方法

| 方法 | 说明 |
|:-----|:-----|
| `fixed_ratio` | 固定比例法，每次买入占总资产的固定百分比 |
| `kelly` | Kelly公式，根据胜率和盈亏比计算最优仓位 |
| `atr_based` | ATR自适应法，根据波动率动态调整仓位 |

---

## FAQ

### Q1: 安装时出现依赖冲突怎么办？

建议使用虚拟环境隔离：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

如仍有冲突，尝试：

```bash
pip install --upgrade pip
pip install -e ".[dev]" --force-reinstall
```

### Q2: AKShare 获取数据失败？

常见原因：
1. **网络问题**: AKShare 依赖网络爬虫，确保网络通畅
2. **股票代码错误**: A股代码为6位数字（如 000001、600519）
3. **日期范围**: 确保日期范围内有交易日数据
4. **接口变化**: AKShare接口偶尔变动，尝试 `pip install --upgrade akshare`

### Q3: 回测结果显示收益率为 0？

可能原因：
1. **数据不足**: 数据量过少，策略未产生交易信号
2. **参数不当**: 如均线窗口大于数据天数
3. **资金不足**: 初始资金过少，无法买入1手（100股）

### Q4: 如何选择合适的策略？

根据市场环境选择：
- **趋势市**: 均线交叉、海龟交易、MACD
- **震荡市**: 布林带、均值回归、RSI
- **不确定**: 复合策略（多策略投票）

### Q5: 参数优化结果很好，但实盘效果差？

这是典型的**过拟合**问题，建议：
1. 使用滚动前进分析（Walk-Forward）验证参数稳定性
2. 增加测试数据的时间跨度
3. 减少优化参数的数量和候选值
4. 关注夏普比率而非总收益率

### Q6: 如何添加自己的策略？

参考 [策略开发指南](STRATEGY_GUIDE.md) 的详细步骤。核心流程：
1. 继承 `Strategy` 基类
2. 实现 `generate_signals()` 方法
3. 使用 `@StrategyRegistry.register("name")` 注册
4. 在 `__init__.py` 中导入

### Q7: 机器学习策略需要安装额外依赖吗？

不需要。所有ML模型（决策树、随机森林、GBDT、LSTM）均使用纯NumPy实现，无需安装 scikit-learn 或 tensorflow。

### Q8: 如何导出回测报告？

```bash
# 文本报告
quant report -st ma_crossover -s 000001

# CSV格式（含交易记录和净值曲线）
quant report -st ma_crossover -s 000001 -f csv -o ./output
```

### Q9: 支持实盘交易吗？

交易引擎模块 (`trading/`) 提供了 `LiveExecutor` 接口框架。当前已实现模拟交易执行器 (`SimulatedExecutor`)。实盘交易执行器需要对接券商API，用户可基于 `LiveExecutor` 接口自行实现。

### Q10: 回测速度慢怎么优化？

1. **缩小日期范围**: 先用短时间段验证逻辑
2. **启用数据缓存**: 确保 `cache_enabled: true`
3. **减少参数组合**: 参数优化时先粗搜后细搜
4. **使用更少的指标**: 只添加策略真正需要的指标

### Q11: 如何同时回测多只股票？

使用 `PortfolioBacktester`：

```python
from quant_trading.backtest.portfolio_backtest import PortfolioBacktester

backtester = PortfolioBacktester(config)
result = backtester.run(
    strategy=strategy,
    data_dict={"000001": df1, "600519": df2},
    weights="equal",
)
```

### Q12: T+1规则是如何实现的？

系统在 `PortfolioManager` 中实现了完整的T+1规则：
- 买入当日：`available_quantity = 0`（不可卖出）
- 次日开盘：自动将昨日买入的数量转入可卖数量
- 止损止盈检查也会跳过当日买入的持仓

### Q13: 数据支持哪些格式的导出？

通过 `DataExporter` 支持三种格式：
- **CSV**: 通用表格格式，Excel可直接打开
- **Excel**: `.xlsx` 格式（需安装 openpyxl）
- **JSON**: 结构化数据，方便程序读取

### Q14: 如何查看策略的所有可用参数？

```bash
# CLI方式
quant strategies

# Python方式
from quant_trading.strategy import StrategyRegistry
strategy = StrategyRegistry.get("ma_crossover")
print(strategy.get_params())
```

### Q15: 如何自定义技术指标？

继承 `Indicator` 基类：

```python
from quant_trading.indicators.base import Indicator

class MyIndicator(Indicator):
    def __init__(self, window: int = 14):
        self.window = window

    @property
    def name(self) -> str:
        return f"my_indicator_{self.window}"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def calculate(self, data):
        self._validate(data)
        data[self.name] = data["close"].rolling(self.window).mean()
        return data
```

### Q16: 日志输出太多/太少怎么调？

```bash
# CLI 开启详细日志
quant backtest -st ma_crossover -s 000001 -v

# Python 代码中调整日志级别
import logging
logging.getLogger("quant_trading").setLevel(logging.DEBUG)   # 详细
logging.getLogger("quant_trading").setLevel(logging.WARNING)  # 安静
```
