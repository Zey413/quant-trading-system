# 系统架构文档

本文档详细描述 A 股量化交易系统的整体架构设计、各模块职责、数据流向以及扩展方法。

---

## 目录

- [系统架构总览](#系统架构总览)
- [各模块职责](#各模块职责)
- [数据流](#数据流)
- [可插拔架构说明](#可插拔架构说明)
- [A 股交易规则实现细节](#a-股交易规则实现细节)
- [扩展指南](#扩展指南)

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI 层 (Click + Rich)                    │
│           quant fetch | quant backtest | quant plot              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       核心层 (core)                              │
│     AppConfig │ Signal │ Order │ Position │ Portfolio            │
│     BrokerConfig │ RiskConfig │ BacktestConfig                  │
│     枚举: SignalType │ OrderSide │ OrderStatus │ ...             │
└────────┬────────────────┬────────────────┬──────────────────────┘
         │                │                │
         ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────┐
│   数据层     │ │   策略层     │ │         回测引擎层           │
│   (data)     │ │  (strategy)  │ │        (backtest)            │
│              │ │              │ │                              │
│ DataSource   │ │ Strategy     │ │  BacktestEngine              │
│  (ABC)       │ │  (ABC)       │ │    ├── SimulatedBroker       │
│              │ │              │ │    ├── PortfolioManager       │
│ ┌──────────┐ │ │ ┌──────────┐ │ │    └── PerformanceMetrics    │
│ │ AKShare  │ │ │ │ MA交叉   │ │ │                              │
│ │ Tushare  │ │ │ │ RSI      │ │ │         ┌─────────┐          │
│ │ (可扩展) │ │ │ │ MACD     │ │ │         │ 风控层  │          │
│ └──────────┘ │ │ │ 布林带   │ │ │         │ (risk)  │          │
│              │ │ │ 复合策略 │ │ │         │         │          │
│ DataManager  │ │ │ (可扩展) │ │ │  ┌──────┤RiskMgr  │          │
│ (Facade)     │ │ └──────────┘ │ │  │      │PosSizer │          │
│              │ │              │ │  │      │StopLoss │          │
│ DataSource   │ │ Strategy     │ │  │      └─────────┘          │
│ Registry     │ │ Registry     │ │  │                            │
└──────────────┘ └──────────────┘ └──┼────────────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │     指标层            │
                          │   (indicators)        │
                          │                      │
                          │ ┌───────┬──────────┐ │
                          │ │ SMA   │ RSI      │ │
                          │ │ EMA   │ KDJ      │ │
                          │ │ MACD  │ Bollinger│ │
                          │ │       │ ATR/OBV  │ │
                          │ └───────┴──────────┘ │
                          └──────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │     可视化层          │
                          │  (visualization)      │
                          │                      │
                          │ ChartGenerator       │
                          │  ├── K线图           │
                          │  ├── 交易信号图       │
                          │  ├── 净值曲线         │
                          │  ├── 回撤图           │
                          │  └── 月度收益热力图   │
                          └──────────────────────┘
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **可插拔架构** | 数据源、策略均通过注册表 + 抽象基类实现，新增实现零改动核心代码 |
| **关注点分离** | 数据获取、策略计算、订单执行、风控验证各自独立 |
| **门面模式** | `DataManager` 封装数据源选择、缓存管理，对外提供简单接口 |
| **协议/ABC** | 回测引擎通过 Python Protocol 定义策略接口，支持鸭子类型 |
| **配置驱动** | 所有参数通过 YAML + Pydantic 统一管理，运行时可覆盖 |

---

## 各模块职责

### core — 核心层

**文件：** `config.py`, `models.py`, `enums.py`

核心层定义了系统中所有共享的数据结构和配置：

| 组件 | 职责 |
|------|------|
| `AppConfig` | 全局配置管理（Pydantic），从 YAML 文件加载 |
| `BrokerConfig` | A 股交易规则参数（佣金率、印花税率、滑点、T+1 等） |
| `Signal` | 交易信号数据类（日期、股票代码、方向、强度） |
| `Order` | 交易订单数据类（含成交价、佣金、税费） |
| `Position` | 持仓数据类（持仓量、可卖量、均价、浮盈） |
| `Portfolio` | 投资组合数据类（现金、持仓字典、总资产） |
| `TradeRecord` | 成交记录（含平仓盈亏） |
| 枚举类型 | `SignalType`, `OrderSide`, `OrderStatus`, `AdjustType` 等 |

**设计要点：**

- 使用 `@dataclass` 定义数据模型，简洁且有类型提示
- `Portfolio` 的 `total_value`、`market_value` 等为计算属性（`@property`）
- 配置使用 Pydantic `BaseModel`，提供自动验证和默认值

---

### data — 数据层

**文件：** `base.py`, `manager.py`, `akshare_source.py`, `tushare_source.py`

```
DataManager (门面)
    │
    ├── 缓存管理 (Parquet 文件)
    │
    └── DataSourceRegistry
            │
            ├── AKShareDataSource  (@register("akshare"))
            ├── TushareDataSource  (@register("tushare"))
            └── 自定义数据源...    (@register("xxx"))
```

| 组件 | 职责 |
|------|------|
| `DataSource` (ABC) | 数据源抽象接口：`fetch_daily()`, `fetch_stock_list()` |
| `DataSourceRegistry` | 装饰器注册表，按名称管理所有数据源实现 |
| `DataManager` | 门面模式，封装数据源选择、Parquet 缓存、列名标准化 |
| `AKShareDataSource` | AKShare 免费数据源实现 |
| `TushareDataSource` | Tushare Pro 数据源实现 |

**数据标准化流程：**

1. 数据源返回原始 DataFrame
2. `DataManager._standardize()` 统一列名为 `date, open, high, low, close, volume`
3. 转换数据类型（datetime、float、int64）
4. 按日期升序排列
5. 写入 Parquet 缓存（下次直接读取）

---

### indicators — 指标层

**文件：** `base.py`, `trend.py`, `oscillator.py`, `volatility.py`, `volume.py`

| 类别 | 指标 | 输出列 |
|------|------|--------|
| 趋势 | `SMA` | `sma_{window}` |
| 趋势 | `EMA` | `ema_{window}` |
| 趋势 | `MACD` | `macd`, `macd_signal`, `macd_hist` |
| 振荡 | `RSI` | `rsi_{period}` |
| 振荡 | `KDJ` | `k`, `d`, `j` |
| 波动率 | `BollingerBands` | `bb_upper`, `bb_middle`, `bb_lower` |
| 波动率 | `ATR` | `atr_{window}` |
| 成交量 | `OBV` | `obv` |
| 成交量 | `VWAP` | `vwap` |

**设计要点：**

- 所有指标继承 `Indicator` 抽象基类
- 必须实现 `name`、`required_columns`、`calculate` 三个接口
- `calculate()` 原地添加指标列并返回 DataFrame
- `_validate()` 方法检查输入数据是否包含所需列

---

### strategy — 策略层

**文件：** `base.py`, `ma_crossover.py`, `rsi_strategy.py`, `macd_strategy.py`, `bollinger_strategy.py`, `composite.py`

```
Strategy (ABC)
    │
    ├── MACrossoverStrategy   — 均线交叉（金叉买/死叉卖）
    ├── RSIStrategy           — RSI 超买超卖
    ├── MACDStrategy          — MACD 金叉死叉
    ├── BollingerStrategy     — 布林带触轨回归
    └── CompositeStrategy     — 多策略投票组合
```

**两种策略接口：**

回测引擎同时支持两种策略接口：

| 接口 | 方法 | 适用场景 |
|------|------|----------|
| 批量方式 | `generate_signals(data) -> DataFrame` | 所有内置策略使用此方式 |
| 逐日方式 | `on_bar(date, row, snapshot) -> Signal` | 需要组合状态信息时使用 |

引擎自动检测策略接口类型并选择对应执行路径。

**CompositeStrategy 投票机制：**

| 投票模式 | 规则 |
|----------|------|
| `majority` | 超过半数策略同意时发出信号 |
| `unanimous` | 全部策略一致时发出信号 |
| `any` | 任一策略发出信号即生效（冲突时 buy 优先） |

**StrategyRegistry 注册表：**

```python
@StrategyRegistry.register("ma_crossover")
class MACrossoverStrategy(Strategy):
    ...

# 运行时按名称获取
strategy = StrategyRegistry.get("ma_crossover", short_window=5)
```

---

### backtest — 回测引擎层

**文件：** `engine.py`, `broker.py`, `portfolio.py`, `metrics.py`

```
BacktestEngine
    │
    ├── SimulatedBroker     — 订单验证 + 执行（A 股规则）
    ├── PortfolioManager    — 持仓管理 + T+1 + 盈亏计算
    ├── RiskManager         — 风控验证 + 止损止盈检查
    └── PerformanceMetrics  — 绩效指标计算
```

**回测引擎单日处理流程：**

```
每个交易日 (current_date):
    │
    ├── 1. new_trading_day()    ← T+1：前日买入转为可卖
    │
    ├── 2. update_prices()      ← 更新持仓当前价格
    │
    ├── 3. check_risk_exits()   ← 遍历持仓检查止损/止盈
    │   └── 如触发 → 生成卖出信号 → 执行卖出
    │
    ├── 4. process_signal()     ← 获取策略信号
    │   ├── on_bar() 或 signal 列
    │   ├── 风控验证 validate_signal()
    │   ├── 计算买入数量 calc_buy_quantity()
    │   └── 创建订单 → 验证 → 执行 → 更新组合
    │
    └── 5. 记录净值 equity_curve_data[date] = total_value
```

---

### risk — 风控层

**文件：** `manager.py`, `position_sizer.py`, `stop_loss.py`

```
RiskManager (统一入口)
    │
    ├── PositionSizer      — 仓位计算
    │   ├── fixed_ratio    — 固定比例法
    │   ├── kelly          — Kelly 公式（半 Kelly）
    │   └── atr_based      — ATR 自适应法
    │
    └── StopLossManager    — 止损止盈
        ├── check_stop_loss()    — 固定百分比止损
        └── check_take_profit()  — 固定百分比止盈
```

**风控检查流程：**

```
买入信号到达
    │
    ├── 1. validate_signal()
    │   ├── 检查单只持仓比例 <= max_position_pct
    │   └── 检查总持仓比例 <= max_total_position_pct
    │
    ├── 2. calculate_order_quantity()
    │   ├── 根据配置选择仓位算法
    │   ├── 应用信号强度调整
    │   ├── 应用最大持仓限制
    │   ├── 向下取整到 lot_size
    │   └── 确保不超过可用现金
    │
    └── 3. (每日开盘前) check_exits()
        └── 遍历持仓 → 检查止损/止盈 → 生成退出信号
```

---

### visualization — 可视化层

**文件：** `charts.py`

| 图表方法 | 说明 |
|----------|------|
| `plot_candlestick()` | K 线图（红涨绿跌 — 中国标准）+ 成交量 |
| `plot_signals()` | 收盘价折线 + 均线 + 买卖信号标注 |
| `plot_equity_curve()` | 净值曲线 + 回撤区域填充 |
| `plot_drawdown()` | 回撤百分比随时间变化 + 最大回撤标注 |
| `plot_monthly_returns()` | 月度收益热力图（年 × 月矩阵） |
| `plot_backtest_report()` | 四合一报告面板 |

**颜色方案：** 遵循中国股市标准 — 红色上涨、绿色下跌。

---

### cli — 命令行层

**文件：** `main.py`

| 命令 | 说明 |
|------|------|
| `quant fetch -s 000001` | 获取股票日线数据并表格展示 |
| `quant backtest -st ma_crossover -s 000001` | 运行策略回测 |
| `quant strategies` | 列出所有可用策略及参数 |
| `quant plot -s 000001 -st rsi` | 生成 K 线 / 信号 / 全部图表 |

**技术栈：** Click（命令解析）+ Rich（终端美化输出，表格/面板/颜色）

---

## 数据流

### 完整回测数据流

```
用户输入 (CLI / Python API)
    │
    │  symbol="000001", strategy="ma_crossover"
    ▼
┌─────────────────┐
│   DataManager   │── cache hit? ──→ 读取 Parquet 缓存
│                 │                          │
│   cache miss    │                          │
│       │         │                          ▼
│       ▼         │               标准化 DataFrame
│  DataSource     │──────────────→  (date, OHLCV)
│  .fetch_daily() │
└─────────────────┘
         │
         ▼ DataFrame
┌─────────────────┐
│    Strategy     │
│.generate_signals│──→ 添加 signal 列 ("buy"/"sell"/"hold")
│   ()            │    + 指标列 (sma_5, sma_20, ...)
└─────────────────┘
         │
         ▼ DataFrame with signals
┌─────────────────────────────────────────┐
│           BacktestEngine.run()          │
│                                         │
│  for each trading_day:                  │
│    ├─ update portfolio prices           │
│    ├─ check stop-loss / take-profit     │
│    ├─ read signal → validate → order    │
│    ├─ SimulatedBroker.execute_order()   │
│    ├─ PortfolioManager.process_buy/sell │
│    └─ record equity_curve               │
│                                         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│       PerformanceMetrics.calculate()    │
│                                         │
│  输入: equity_curve + trades            │
│                                         │
│  输出: PerformanceResult                │
│    ├─ total_return, annualized_return   │
│    ├─ max_drawdown, volatility          │
│    ├─ sharpe_ratio, sortino, calmar     │
│    ├─ win_rate, profit_loss_ratio       │
│    └─ equity_curve, drawdown_series     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           ChartGenerator                │
│                                         │
│  plot_backtest_report()                 │
│    ├─ 交易信号图                         │
│    ├─ 净值曲线                           │
│    ├─ 回撤图                             │
│    └─ 月度收益热力图                     │
└─────────────────────────────────────────┘
```

### 订单执行数据流

```
Signal(buy, 000001, 10.0)
    │
    ├─ RiskManager.validate_signal()  ──→ 拒绝? return
    │
    ├─ RiskManager.calculate_order_quantity()
    │     └─ PositionSizer → 1000 股
    │
    ├─ Engine._calc_buy_quantity()
    │     └─ 取整到 lot_size → 1000 股
    │
    ├─ Order(buy, 000001, 1000, 10.0)
    │
    ├─ SimulatedBroker.validate_order()
    │     ├─ 检查资金充足
    │     └─ 检查手数合规
    │
    ├─ SimulatedBroker.execute_order()
    │     ├─ 计算滑点 → filled_price = 10.01
    │     ├─ 计算佣金 → max(10.01*1000*0.0003, 5) = 5.0
    │     ├─ 计算印花税 → 0 (买入不收)
    │     └─ status = FILLED
    │
    └─ PortfolioManager.process_buy()
          ├─ cash -= 10.01*1000 + 5.0
          ├─ 新建 Position(qty=1000, available=0)  ← T+1
          └─ _pending_available["000001"] = 1000
```

---

## 可插拔架构说明

### 注册表模式

系统中数据源和策略均采用**装饰器注册表**模式，实现插件式扩展：

```python
# 1. 定义注册表（类级别字典）
class StrategyRegistry:
    _strategies: dict[str, type[Strategy]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(strategy_cls):
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Strategy:
        return cls._strategies[name](**kwargs)

# 2. 通过装饰器注册
@StrategyRegistry.register("ma_crossover")
class MACrossoverStrategy(Strategy): ...

# 3. 运行时按名称获取
strategy = StrategyRegistry.get("ma_crossover", short_window=5)
```

### 扩展新实现的步骤

1. **继承抽象基类**（`DataSource` 或 `Strategy`）
2. **使用 `@register` 装饰器**注册到对应注册表
3. **在 `__init__.py` 中导入**新模块以触发注册
4. 无需修改引擎、CLI 或其他模块的代码

### 策略接口的灵活性

回测引擎同时支持两种策略接口（运行时自动检测）：

```python
# 方式 1: 批量信号生成（推荐，所有内置策略使用此方式）
class MyStrategy(Strategy):
    def generate_signals(self, data: DataFrame) -> DataFrame:
        data["signal"] = "hold"
        ...
        return data

# 方式 2: 逐日信号（需要运行时组合状态时使用）
class MyOnBarStrategy:
    def on_bar(self, date, row, snapshot) -> Signal | None:
        if snapshot["cash"] > 100000:
            return Signal(...)
        return None
```

---

## A 股交易规则实现细节

### T+1 规则

**规则：** 当日买入的股票，次日才能卖出。

**实现位置：** `PortfolioManager`

```
买入时:
    Position.available_quantity = 0           ← 当日不可卖
    _pending_available[symbol] += quantity    ← 记录待转入

每个新交易日开盘前 (new_trading_day()):
    for symbol, qty in _pending_available:
        position.available_quantity += qty    ← 转入可卖
    _pending_available.clear()
```

**风控联动：** `RiskManager.check_exits()` 中，`available_quantity <= 0` 的持仓不生成退出信号。

### 最小交易单位（手）

**规则：** A 股交易必须以 100 股（1 手）为单位。

**实现位置：** `SimulatedBroker._round_to_lot_size()`

```python
quantity = (quantity // lot_size) * lot_size  # 向下取整
# 不足 100 股的订单被拒绝 → OrderStatus.REJECTED
```

### 佣金计算

**规则：** 佣金 = max(成交额 × 佣金率, 最低佣金)

**默认参数：**
- 佣金率：万三（0.0003）
- 最低佣金：5 元

```python
commission = max(turnover * commission_rate, min_commission)
```

### 印花税

**规则：** 仅卖出时收取，税率千分之一。

```python
tax = turnover * stamp_tax_rate if side == "sell" else 0.0
```

### 滑点模拟

**规则：** 买入价格上浮，卖出价格下浮。

```python
if order.side == BUY:
    filled_price = current_price * (1 + slippage)   # 买贵
else:
    filled_price = current_price * (1 - slippage)   # 卖便宜
```

### 涨跌停限制

当前版本暂未在 Broker 层硬编码涨跌停限制（±10% / ±5%）。合成数据和真实数据中的价格变动自然反映了涨跌停对行情的影响。未来版本可在 `SimulatedBroker` 中添加涨跌停验证。

---

## 扩展指南

### 添加新策略

详见 [CONTRIBUTING.md](../CONTRIBUTING.md#如何添加新策略)。

核心步骤：
1. 创建 `src/quant_trading/strategy/xxx.py`
2. 继承 `Strategy`，实现 `generate_signals()`
3. 用 `@StrategyRegistry.register("xxx")` 注册
4. 在 `__init__.py` 中导入
5. 编写测试

### 添加新数据源

核心步骤：
1. 创建 `src/quant_trading/data/xxx_source.py`
2. 继承 `DataSource`，实现 `fetch_daily()` 和 `fetch_stock_list()`
3. 用 `@DataSourceRegistry.register("xxx")` 注册
4. 在 `__init__.py` 中导入
5. 返回的 DataFrame 必须包含标准列：`date, open, high, low, close, volume`

### 添加新指标

核心步骤：
1. 在对应分类文件中添加（`trend.py` / `oscillator.py` / `volatility.py` / `volume.py`）
2. 继承 `Indicator`，实现 `name`、`required_columns`、`calculate`
3. `calculate()` 在 DataFrame 上添加指标列并返回

### 添加新 CLI 命令

在 `src/quant_trading/cli/main.py` 中：

```python
@cli.command()
@click.option('--param', '-p', required=True, help='参数说明')
@click.pass_context
def my_command(ctx, param):
    """命令说明"""
    config = ctx.obj['config']
    # 实现逻辑...
```

### 添加新仓位管理方法

1. 在 `PositionSizer` 中添加新的 `_xxx_sizing()` 方法
2. 在 `calculate_quantity()` 的分支中添加对应判断
3. 在 `PositionSizingMethod` 枚举中添加新值
4. 在 `RiskConfig` 中添加相关配置字段

### 添加新止损方法

1. 在 `StopLossManager` 中添加新的检查方法
2. 在 `should_exit()` 中集成新检查
3. 在 `StopLossMethod` 枚举中添加新值
4. 在 `RiskConfig` 中添加相关配置字段
