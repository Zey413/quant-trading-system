# API 参考文档

本文档详细列出量化交易系统所有公开模块、类、方法的接口说明。

---

## 目录

- [core — 核心层](#core--核心层)
- [data — 数据层](#data--数据层)
- [indicators — 指标层](#indicators--指标层)
- [strategy — 策略层](#strategy--策略层)
- [backtest — 回测引擎层](#backtest--回测引擎层)
- [risk — 风险管理层](#risk--风险管理层)
- [ml — 机器学习层](#ml--机器学习层)
- [trading — 交易引擎层](#trading--交易引擎层)
- [visualization — 可视化层](#visualization--可视化层)
- [cli — 命令行层](#cli--命令行层)

---

## core — 核心层

### `AppConfig`

```python
from quant_trading.core.config import AppConfig
```

全局配置管理器，基于 Pydantic BaseModel。

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `data_source` | `DataSourceConfig` | 数据源配置 |
| `broker` | `BrokerConfig` | 交易规则配置 |
| `backtest` | `BacktestConfig` | 回测参数配置 |
| `risk` | `RiskConfig` | 风控参数配置 |

#### 方法

**`AppConfig.from_yaml(path: str | Path) -> AppConfig`**

从 YAML 文件加载配置。文件不存在时返回默认配置。

```python
config = AppConfig.from_yaml("config/default.yaml")
```

**`config.to_yaml(path: str | Path) -> None`**

保存配置到 YAML 文件。

```python
config.to_yaml("config/custom.yaml")
```

**`config.validate() -> list[str]`**

手动校验所有约束，返回错误信息列表。空列表表示全部通过。

```python
errors = config.validate()
if errors:
    for e in errors:
        print(f"配置错误: {e}")
```

---

### `DataSourceConfig`

```python
from quant_trading.core.config import DataSourceConfig
```

| 属性 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `default_source` | `str` | `"akshare"` | 默认数据源 |
| `tushare_token` | `str` | `""` | Tushare Pro Token |
| `cache_dir` | `str` | `"data_cache"` | 缓存目录 |
| `cache_enabled` | `bool` | `True` | 是否启用缓存 |

---

### `BrokerConfig`

```python
from quant_trading.core.config import BrokerConfig
```

| 属性 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `commission_rate` | `float` | `0.0003` | 佣金率（万三），范围 0~0.01 |
| `min_commission` | `float` | `5.0` | 最低佣金（元） |
| `stamp_tax_rate` | `float` | `0.001` | 印花税率（千一，仅卖出） |
| `slippage` | `float` | `0.001` | 滑点（千一） |
| `lot_size` | `int` | `100` | 最小交易单位（股） |
| `trading_days_per_year` | `int` | `244` | 年交易日数 |
| `t_plus_1` | `bool` | `True` | T+1 规则 |

---

### `BacktestConfig`

```python
from quant_trading.core.config import BacktestConfig
```

| 属性 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `initial_capital` | `float` | `1000000.0` | 初始资金（元），必须 > 0 |
| `start_date` | `str` | `"2023-01-01"` | 起始日期 |
| `end_date` | `str` | `"2025-12-31"` | 结束日期 |
| `benchmark` | `str` | `"000300"` | 基准指数 |
| `risk_free_rate` | `float` | `0.025` | 无风险利率 |

---

### `RiskConfig`

```python
from quant_trading.core.config import RiskConfig
```

| 属性 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `max_position_pct` | `float` | `0.3` | 单只最大仓位比例 |
| `max_total_position_pct` | `float` | `0.8` | 最大总仓位比例 |
| `stop_loss_pct` | `float` | `0.05` | 止损百分比 |
| `take_profit_pct` | `float` | `0.15` | 止盈百分比 |
| `position_sizing_method` | `str` | `"fixed_ratio"` | 仓位算法 |
| `fixed_ratio` | `float` | `0.1` | 固定比例仓位 |

---

### 数据模型

```python
from quant_trading.core.models import Signal, Order, Position, Portfolio, TradeRecord
```

#### `Signal`

交易信号数据类。

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `date` | `date` | 信号日期 |
| `symbol` | `str` | 股票代码 |
| `signal_type` | `SignalType` | 信号类型（BUY/SELL/HOLD） |
| `price` | `float` | 参考价格 |
| `strength` | `float` | 信号强度 0~1 |

#### `Order`

交易订单数据类。

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `order_id` | `str` | 订单ID |
| `symbol` | `str` | 股票代码 |
| `side` | `OrderSide` | 买/卖方向 |
| `quantity` | `int` | 数量 |
| `price` | `float` | 委托价格 |
| `filled_price` | `float` | 成交价格 |
| `commission` | `float` | 佣金 |
| `tax` | `float` | 印花税 |
| `status` | `OrderStatus` | 订单状态 |

#### `Position`

持仓数据类。

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `symbol` | `str` | 股票代码 |
| `quantity` | `int` | 持仓数量 |
| `available_quantity` | `int` | 可卖数量（T+1） |
| `avg_cost` | `float` | 持仓均价 |
| `current_price` | `float` | 当前价格 |
| `unrealized_pnl` | `float` | 浮动盈亏 |

#### `Portfolio`

投资组合数据类。

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `cash` | `float` | 可用现金 |
| `positions` | `dict` | 持仓字典 {symbol: Position} |
| `total_value` | `float` | 总资产（计算属性） |
| `market_value` | `float` | 持仓市值（计算属性） |

#### `TradeRecord`

成交记录数据类。

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `date` | `date` | 成交日期 |
| `symbol` | `str` | 股票代码 |
| `side` | `OrderSide` | 方向 |
| `quantity` | `int` | 数量 |
| `price` | `float` | 成交价 |
| `commission` | `float` | 佣金 |
| `tax` | `float` | 印花税 |
| `pnl` | `float` | 平仓盈亏（仅卖出） |

---

### 枚举类型

```python
from quant_trading.core.enums import SignalType, OrderSide, OrderStatus, AdjustType
```

| 枚举 | 值 | 说明 |
|:-----|:---|:-----|
| `SignalType.BUY` | `"buy"` | 买入信号 |
| `SignalType.SELL` | `"sell"` | 卖出信号 |
| `SignalType.HOLD` | `"hold"` | 持有 |
| `OrderSide.BUY` | `"buy"` | 买入 |
| `OrderSide.SELL` | `"sell"` | 卖出 |
| `OrderStatus.PENDING` | `"pending"` | 待执行 |
| `OrderStatus.FILLED` | `"filled"` | 已成交 |
| `OrderStatus.REJECTED` | `"rejected"` | 被拒绝 |

---

## data — 数据层

### `DataManager`

```python
from quant_trading.data.manager import DataManager
```

数据管理器（门面模式），封装数据源选择和缓存管理。

#### 构造方法

```python
DataManager(config: AppConfig)
```

#### 方法

**`fetch_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame`**

获取股票日线数据。自动使用缓存。

```python
dm = DataManager(config)
df = dm.fetch_daily("000001", "2024-01-01", "2025-01-01")
```

返回标准化 DataFrame，包含 `date, open, high, low, close, volume` 列。

**`set_source(source_name: str) -> None`**

切换数据源。

```python
dm.set_source("tushare")
```

**`source_name -> str`**

当前数据源名称（属性）。

---

### `DataSource` (ABC)

```python
from quant_trading.data.base import DataSource
```

数据源抽象基类。

| 方法 | 返回类型 | 说明 |
|:-----|:---------|:-----|
| `get_name()` | `str` | 数据源名称 |
| `fetch_daily(symbol, start_date, end_date, adjust)` | `DataFrame` | 获取日线数据 |
| `fetch_stock_list()` | `DataFrame` | 获取股票列表 |

---

### `DataSourceRegistry`

```python
from quant_trading.data.base import DataSourceRegistry
```

| 方法 | 说明 |
|:-----|:-----|
| `@register(name)` | 装饰器，注册数据源 |
| `get(name) -> DataSource` | 获取数据源实例 |
| `list_sources() -> list[str]` | 列出所有已注册数据源 |

---

### `DataExporter`

```python
from quant_trading.data.export import DataExporter
```

数据导出工具。所有方法为静态方法。

| 方法 | 说明 |
|:-----|:-----|
| `to_csv(df, filepath)` | 导出为 CSV |
| `to_excel(df, filepath)` | 导出为 Excel（需 openpyxl） |
| `to_json(df, filepath)` | 导出为 JSON（records格式） |

```python
DataExporter.to_csv(df, "output/data.csv")
DataExporter.to_excel(df, "output/data.xlsx")
DataExporter.to_json(df, "output/data.json")
```

---

## indicators — 指标层

### `Indicator` (ABC)

```python
from quant_trading.indicators.base import Indicator
```

指标抽象基类。

| 属性/方法 | 类型 | 说明 |
|:----------|:-----|:-----|
| `name` | `str` (property) | 指标名称 |
| `required_columns` | `list[str]` (property) | 依赖的数据列 |
| `calculate(data)` | `DataFrame` | 计算指标，原地添加列 |
| `_validate(data)` | `None` | 校验输入数据 |

---

### 趋势指标 (`indicators.trend`)

#### `SMA`

简单移动平均线。添加 `sma_{window}` 列。

```python
SMA(window: int = 20, column: str = "close")
```

#### `EMA`

指数移动平均线。添加 `ema_{window}` 列。

```python
EMA(window: int = 20, column: str = "close")
```

#### `MACD`

MACD 指标。添加 `macd`, `macd_signal`, `macd_hist` 列。

```python
MACD(fast: int = 12, slow: int = 26, signal: int = 9)
```

---

### 振荡指标 (`indicators.oscillator`)

#### `RSI`

相对强弱指标。添加 `rsi_{period}` 列。范围 0~100。

```python
RSI(period: int = 14)
```

#### `KDJ`

KDJ 随机指标。添加 `k`, `d`, `j` 列。

```python
KDJ(k_period: int = 9, d_period: int = 3)
```

---

### 波动率指标 (`indicators.volatility`)

#### `BollingerBands`

布林带。添加 `bb_upper`, `bb_middle`, `bb_lower` 列。

```python
BollingerBands(window: int = 20, num_std: float = 2.0)
```

#### `ATR`

平均真实波幅。添加 `atr_{window}` 列。

```python
ATR(window: int = 14)
```

---

### 成交量指标 (`indicators.volume`)

#### `OBV`

能量潮指标。添加 `obv` 列。

```python
OBV()
```

#### `VWAP`

成交量加权平均价格。添加 `vwap` 列。

```python
VWAP()
```

---

### 动量指标 (`indicators.momentum`)

#### `ROC`

变动率。添加 `roc_{period}` 列。

```python
ROC(period: int = 12, column: str = "close")
```

计算方式: `ROC = (close - close_n) / close_n * 100`

#### `WilliamsR`

威廉指标。添加 `williams_r_{period}` 列。范围 -100~0。

```python
WilliamsR(period: int = 14)
```

计算方式: `%R = (highest_high - close) / (highest_high - lowest_low) * -100`

#### `CCI`

顺势指标。添加 `cci_{period}` 列。

```python
CCI(period: int = 20)
```

计算方式: `CCI = (TP - SMA(TP)) / (0.015 * mean_deviation)`

---

### 指标工具 (`indicators.utils`)

**`add_all_indicators(data, config=None) -> DataFrame`**

一键添加所有常用指标。

```python
from quant_trading.indicators.utils import add_all_indicators

data = add_all_indicators(data)
# 默认添加: SMA(5,10,20,60), EMA(12,26), RSI(14), MACD, BB(20),
#           ATR(14), OBV, VWAP, ROC(12), WilliamsR(14), CCI(20)
```

**`calculate_indicator_correlation(data, indicators) -> DataFrame`**

计算指标间的 Pearson 相关性矩阵。

```python
from quant_trading.indicators.utils import calculate_indicator_correlation

corr = calculate_indicator_correlation(data, ["rsi_14", "sma_20", "cci_20"])
```

---

## strategy — 策略层

### `Strategy` (ABC)

```python
from quant_trading.strategy.base import Strategy
```

| 方法 | 返回类型 | 说明 |
|:-----|:---------|:-----|
| `generate_signals(data)` | `DataFrame` | 生成信号（抽象方法） |
| `get_params()` | `dict` | 返回参数字典 |

### `StrategyRegistry`

```python
from quant_trading.strategy.base import StrategyRegistry
```

| 方法 | 说明 |
|:-----|:-----|
| `@register(name)` | 注册装饰器 |
| `get(name, **kwargs) -> Strategy` | 按名称获取策略实例 |
| `list_strategies() -> list[str]` | 列出所有已注册策略 |

```python
# 注册
@StrategyRegistry.register("my_strategy")
class MyStrategy(Strategy): ...

# 获取
strategy = StrategyRegistry.get("my_strategy", param1=10)

# 列表
names = StrategyRegistry.list_strategies()
```

### 内置策略

#### `MACrossoverStrategy`

```python
MACrossoverStrategy(short_window: int = 5, long_window: int = 20, name: str = "")
```

均线交叉策略。短期均线上穿长期均线（金叉）买入，下穿（死叉）卖出。

#### `RSIStrategy`

```python
RSIStrategy(period: int = 14, oversold: float = 30, overbought: float = 70, name: str = "")
```

RSI 超买超卖策略。

#### `MACDStrategy`

```python
MACDStrategy(fast: int = 12, slow: int = 26, signal: int = 9, name: str = "")
```

MACD 金叉死叉策略。

#### `BollingerStrategy`

```python
BollingerStrategy(window: int = 20, num_std: float = 2.0, name: str = "")
```

布林带触轨回归策略。

#### `DualThrustStrategy`

```python
DualThrustStrategy(lookback: int = 4, k1: float = 0.5, k2: float = 0.5, name: str = "")
```

Dual Thrust 动量突破策略。

#### `MeanReversionStrategy`

```python
MeanReversionStrategy(window: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.0, name: str = "")
```

均值回归（Z-Score）策略。

#### `TurtleStrategy`

```python
TurtleStrategy(entry_period: int = 20, exit_period: int = 10, atr_period: int = 20, name: str = "")
```

海龟交易法策略。

#### `CompositeStrategy`

```python
CompositeStrategy(strategies: list[Strategy], voting: str = "majority", name: str = "")
```

复合策略。voting 可选: `"majority"` / `"unanimous"` / `"any"`。

---

## backtest — 回测引擎层

### `BacktestEngine`

```python
from quant_trading.backtest.engine import BacktestEngine
```

#### 构造方法

```python
BacktestEngine(config: AppConfig)
```

#### 属性

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `config` | `AppConfig` | 全局配置 |
| `broker` | `SimulatedBroker` | 模拟券商 |
| `portfolio_manager` | `PortfolioManager` | 组合管理器 |
| `trades` | `list[TradeRecord]` | 成交记录 |

#### 方法

**`run(strategy, data, symbol) -> PerformanceResult`**

运行单标的回测。

```python
engine = BacktestEngine(config)
result = engine.run(strategy, data, symbol="000001")
```

参数:
- `strategy`: 策略实例（实现 `generate_signals` 或 `on_bar`）
- `data`: 行情 DataFrame
- `symbol`: 股票代码

返回: `PerformanceResult` 绩效结果对象。

---

### `PerformanceResult`

```python
from quant_trading.backtest.metrics import PerformanceResult
```

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `total_return` | `float` | 总收益率 |
| `annualized_return` | `float` | 年化收益率 |
| `max_drawdown` | `float` | 最大回撤 |
| `max_drawdown_duration` | `int` | 最大回撤持续天数 |
| `volatility` | `float` | 年化波动率 |
| `sharpe_ratio` | `float` | 夏普比率 |
| `sortino_ratio` | `float` | Sortino比率 |
| `calmar_ratio` | `float` | Calmar比率 |
| `total_trades` | `int` | 总交易次数 |
| `win_rate` | `float` | 胜率 |
| `profit_loss_ratio` | `float` | 盈亏比 |
| `avg_win` | `float` | 平均盈利 |
| `avg_loss` | `float` | 平均亏损 |
| `equity_curve` | `pd.Series` | 净值曲线 |
| `drawdown_series` | `pd.Series` | 回撤序列 |

**`summary() -> str`**

返回格式化的绩效摘要文本。

---

### `PerformanceMetrics`

```python
from quant_trading.backtest.metrics import PerformanceMetrics
```

绩效指标计算器（静态方法类）。

**`calculate(equity_curve, trades, config) -> PerformanceResult`**

从净值曲线和交易记录计算全部绩效指标。

---

### `GridSearchOptimizer`

```python
from quant_trading.backtest.optimizer import GridSearchOptimizer
```

#### 构造方法

```python
GridSearchOptimizer(config: AppConfig)
```

#### 方法

**`optimize(strategy_name, param_grid, data, symbol, metric) -> OptimizationResult`**

执行网格搜索优化。

```python
result = optimizer.optimize(
    strategy_name="ma_crossover",
    param_grid={"short_window": [3, 5, 10], "long_window": [15, 20, 30]},
    data=data,
    symbol="000001",
    metric="sharpe_ratio",
)
```

参数:
- `strategy_name`: 已注册的策略名称
- `param_grid`: 参数网格 `{param: [value1, value2, ...]}`
- `data`: 行情 DataFrame
- `symbol`: 股票代码
- `metric`: 优化指标名称

返回: `OptimizationResult`（含 `best_params`, `best_metric_value`, `all_results`）

---

### `WalkForwardAnalyzer`

```python
from quant_trading.backtest.walk_forward import WalkForwardAnalyzer
```

#### 构造方法

```python
WalkForwardAnalyzer(config: AppConfig)
```

#### 方法

**`run(strategy_name, param_grid, data, symbol, train_size, test_size, step_size, optimize_metric) -> WalkForwardResult`**

执行滚动前进分析。

```python
result = analyzer.run(
    strategy_name="ma_crossover",
    param_grid={"short_window": [3, 5, 10], "long_window": [15, 20, 30]},
    data=df,
    symbol="000001",
    train_size=120,
    test_size=60,
    step_size=60,
    optimize_metric="sharpe_ratio",
)
```

参数:
- `train_size`: 训练期交易日数
- `test_size`: 测试期交易日数
- `step_size`: 每次前进交易日数

返回: `WalkForwardResult`

---

### `WalkForwardResult`

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `windows` | `list[WalkForwardWindow]` | 各窗口详细结果 |
| `overall_test_equity` | `pd.Series` | 拼接后的测试期净值曲线 |
| `overall_performance` | `PerformanceResult` | 整体测试期绩效 |

**`summary() -> str`**: 格式化摘要。

### `WalkForwardWindow`

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `train_start` | `str` | 训练期开始日期 |
| `train_end` | `str` | 训练期结束日期 |
| `test_start` | `str` | 测试期开始日期 |
| `test_end` | `str` | 测试期结束日期 |
| `best_params` | `dict` | 训练期优化出的最优参数 |
| `train_result` | `PerformanceResult` | 训练期回测绩效 |
| `test_result` | `PerformanceResult` | 测试期回测绩效 |

---

### `PortfolioBacktester`

```python
from quant_trading.backtest.portfolio_backtest import PortfolioBacktester
```

#### 构造方法

```python
PortfolioBacktester(config: AppConfig)
```

#### 方法

**`run(strategy, data_dict, weights) -> PortfolioBacktestResult`**

执行多股票组合回测。

```python
result = backtester.run(
    strategy=strategy,
    data_dict={"000001": df1, "600519": df2},
    weights="equal",  # 或 {"000001": 0.6, "600519": 0.4}
)
```

参数:
- `strategy`: 策略实例或策略工厂（callable）
- `data_dict`: `{symbol: DataFrame}` 多标的数据
- `weights`: `"equal"` 或自定义权重字典

### `PortfolioBacktestResult`

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `overall` | `PerformanceResult` | 组合整体绩效 |
| `per_stock` | `dict[str, PerformanceResult]` | 个股绩效 |
| `stock_weights` | `dict[str, float]` | 权重 |
| `correlation_matrix` | `pd.DataFrame` | 相关性矩阵 |

---

### `SignalAnalyzer`

```python
from quant_trading.backtest.signal_analyzer import SignalAnalyzer
```

#### 方法

**`analyze(data, forward_periods, signal_column) -> SignalAnalysisResult`**

分析信号质量。

```python
analyzer = SignalAnalyzer()
result = analyzer.analyze(data, forward_periods=[1, 5, 10])
```

参数:
- `data`: 包含 `close` 和 `signal` 列（1=买, -1=卖, 0=无）
- `forward_periods`: 前瞻周期列表
- `signal_column`: 信号列名

**`plot_signal_distribution(data, save_path, signal_column) -> None`**

绘制信号分布图。

### `SignalAnalysisResult`

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `total_signals` | `int` | 信号总数 |
| `buy_signals` | `int` | 买入信号数 |
| `sell_signals` | `int` | 卖出信号数 |
| `avg_return_after_buy` | `float` | 买入后平均收益 |
| `avg_return_after_sell` | `float` | 卖出后平均收益 |
| `buy_win_rate` | `float` | 买入信号正确率 |
| `sell_win_rate` | `float` | 卖出信号正确率 |
| `signal_frequency` | `float` | 信号频率 |
| `avg_holding_period` | `float` | 平均持仓天数 |
| `consecutive_wins` | `int` | 最大连胜 |
| `consecutive_losses` | `int` | 最大连亏 |

---

### `StrategyComparator`

```python
from quant_trading.backtest.comparator import StrategyComparator
```

#### 方法

**`add_strategy(name, **kwargs) -> None`**

添加待对比的策略。

**`compare(data, symbol, rank_by) -> ComparisonResult`**

运行对比。

**`plot_comparison(result, save_path) -> None`**

绘制净值对比图。

---

### `ReportGenerator`

```python
from quant_trading.backtest.report import ReportGenerator
```

所有方法为静态方法。

**`generate_text_report(result, strategy_name, symbol, config, trades) -> str`**

生成文本格式回测报告。

**`generate_csv_trades(trades, filepath) -> None`**

导出交易记录 CSV。

**`generate_equity_csv(equity_curve, filepath) -> None`**

导出净值曲线 CSV。

---

### `SimulatedBroker`

```python
from quant_trading.backtest.broker import SimulatedBroker
```

#### 构造方法

```python
SimulatedBroker(config: BrokerConfig)
```

#### 方法

**`validate_order(order, portfolio) -> bool`**

验证订单合法性（资金、手数等）。

**`execute_order(order, current_price) -> Order`**

执行订单（计算滑点、佣金、印花税）。

---

### `PortfolioManager`

```python
from quant_trading.backtest.portfolio import PortfolioManager
```

#### 构造方法

```python
PortfolioManager(initial_capital: float)
```

#### 方法

| 方法 | 说明 |
|:-----|:-----|
| `process_buy(order)` | 处理买入订单 |
| `process_sell(order)` | 处理卖出订单 |
| `new_trading_day()` | 新交易日处理（T+1转可卖） |
| `update_prices(prices)` | 更新持仓价格 |

---

## risk — 风险管理层

### `RiskManager`

```python
from quant_trading.risk.manager import RiskManager
```

风控管理器统一入口。

| 方法 | 说明 |
|:-----|:-----|
| `validate_signal(signal, portfolio)` | 验证信号（仓位限制） |
| `calculate_order_quantity(signal, portfolio)` | 计算买入数量 |
| `check_exits(portfolio, prices)` | 检查止损/止盈 |

### `PositionSizer`

```python
from quant_trading.risk.position_sizer import PositionSizer
```

仓位管理器。

| 方法 | 说明 |
|:-----|:-----|
| `calculate_quantity(signal, portfolio, method)` | 计算买入数量 |

支持的方法:
- `fixed_ratio`: 固定比例法
- `kelly`: Kelly公式
- `atr_based`: ATR自适应法

### `StopLossManager`

```python
from quant_trading.risk.stop_loss import StopLossManager
```

| 方法 | 说明 |
|:-----|:-----|
| `check_stop_loss(position)` | 检查是否触发止损 |
| `check_take_profit(position)` | 检查是否触发止盈 |

---

## ml — 机器学习层

### `FeatureEngineer`

```python
from quant_trading.ml import FeatureEngineer
```

**`create_features(data) -> DataFrame`**

一键生成技术指标特征。

### `MLModelBase` (ABC)

所有ML模型的抽象基类。

| 方法 | 说明 |
|:-----|:-----|
| `fit(X, y)` | 训练模型 |
| `predict(X) -> ndarray` | 预测 |

### ML模型

| 类 | 构造参数 |
|:---|:---------|
| `DecisionTreeModel` | `max_depth=5, min_samples_split=2` |
| `RandomForestModel` | `n_trees=100, max_depth=10` |
| `LinearModel` | `learning_rate=0.01, n_iterations=1000` |
| `GradientBoostingModel` | `n_estimators=100, learning_rate=0.1, max_depth=3` |
| `LSTMModel` | `hidden_size=64, n_epochs=100` |
| `EnsembleModel` | `models: list[MLModelBase]` |

### `MLPipeline`

```python
from quant_trading.ml import MLPipeline, StandardScaler
```

```python
pipeline = MLPipeline(model=RandomForestModel(), scaler=StandardScaler())
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

| 方法 | 说明 |
|:-----|:-----|
| `fit(X, y)` | 标准化数据 + 训练模型 |
| `predict(X) -> ndarray` | 标准化 + 预测 |

### 标准化器

| 类 | 说明 |
|:---|:-----|
| `StandardScaler` | 零均值单位方差标准化 |
| `MinMaxScaler` | 缩放到 [0, 1] |

| 方法 | 说明 |
|:-----|:-----|
| `fit(X) -> self` | 拟合参数 |
| `transform(X) -> ndarray` | 转换数据 |
| `fit_transform(X) -> ndarray` | 拟合并转换 |

### `ModelEvaluator`

```python
from quant_trading.ml import ModelEvaluator
```

**`evaluate(y_true, y_pred) -> dict`**

返回评估指标字典：`accuracy`, `precision`, `recall`, `f1`。

---

## trading — 交易引擎层

### `ExecutionEngine`

```python
from quant_trading.trading import ExecutionEngine
```

信号执行与撮合引擎。

| 方法 | 说明 |
|:-----|:-----|
| `execute(signal, risk_monitor)` | 执行交易信号 |

### `OrderManager`

```python
from quant_trading.trading import OrderManager
```

订单生命周期管理。

| 方法 | 说明 |
|:-----|:-----|
| `create_order(signal)` | 创建订单 |
| `validate_order(order)` | 验证订单 |
| `cancel_order(order_id)` | 撤销订单 |

### `PositionTracker`

```python
from quant_trading.trading import PositionTracker
```

持仓跟踪与实时盈亏计算。

### `RiskMonitor`

```python
from quant_trading.trading import RiskMonitor, MaxPositionRule, MaxDrawdownRule
```

实时风控监控器。

**`add_rule(rule: RiskRule) -> None`**

添加风控规则。

**`check(order, portfolio) -> bool`**

检查订单是否通过所有风控规则。

### 风控规则

| 规则 | 说明 |
|:-----|:-----|
| `MaxPositionRule(max_pct)` | 单只最大仓位限制 |
| `MaxDrawdownRule(max_dd)` | 最大回撤限制 |
| `DailyLossLimitRule(max_loss)` | 单日最大亏损限制 |
| `ConcentrationRule(max_pct)` | 持仓集中度限制 |

### `TradeLogger`

```python
from quant_trading.trading import TradeLogger
```

交易日志与统计。

---

## visualization — 可视化层

### `ChartGenerator`

```python
from quant_trading.visualization.charts import ChartGenerator
```

所有方法为静态方法或类方法。

| 方法 | 参数 | 说明 |
|:-----|:-----|:-----|
| `plot_candlestick(data, title, save_path)` | data: OHLCV DataFrame | K线图 |
| `plot_signals(data, title, save_path)` | data: 含signal列 | 交易信号图 |
| `plot_equity_curve(equity, title, save_path)` | equity: Series | 净值曲线 |
| `plot_drawdown(dd_series, title, save_path)` | dd_series: Series | 回撤图 |
| `plot_monthly_returns(equity, title, save_path)` | equity: Series | 月度热力图 |
| `plot_backtest_report(data, equity_curve, title, save_path)` | 综合参数 | 四合一面板 |

所有方法的 `save_path` 参数可选。为 `None` 时调用 `plt.show()`，传入路径时保存图片。

---

## cli — 命令行层

### 命令列表

| 命令 | 说明 | 主要参数 |
|:-----|:-----|:---------|
| `quant fetch` | 获取数据 | `-s`, `--start`, `--end`, `--source`, `-n` |
| `quant backtest` | 运行回测 | `-st`, `-s`, `--start`, `--end`, `--capital`, `--save` |
| `quant strategies` | 列出策略 | 无 |
| `quant plot` | 生成图表 | `-s`, `-st`, `-t`, `--save` |
| `quant optimize` | 参数优化 | `-st`, `-s`, `-p`, `-m` |
| `quant report` | 生成报告 | `-st`, `-s`, `-f`, `-o` |

### 全局参数

| 参数 | 说明 |
|:-----|:-----|
| `--config` / `-c` | 配置文件路径（默认 `config/default.yaml`） |
| `--verbose` / `-v` | 开启详细日志 |
