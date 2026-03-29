<div align="center">

# 量化交易系统

**面向中国A股市场的专业量化交易框架**

*可插拔数据源 · 8大内置策略 · 事件驱动回测 · 智能风控 · 参数优化*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)]()

</div>

---

## 功能亮点

| 模块 | 功能 | 说明 |
|:---:|:---|:---|
| **数据层** | 可插拔数据源架构 | AKShare（免费）+ Tushare Pro，装饰器注册表模式，轻松扩展 |
| **策略层** | 8 大内置策略 + 复合策略 | 均线交叉、RSI、MACD、布林带、Dual Thrust、均值回归、海龟交易、复合投票 |
| **回测引擎** | 事件驱动 + A股规则 | 完整支持 T+1、印花税、最小交易单位、滑点模拟 |
| **参数优化** | 网格搜索优化器 | 自动遍历参数空间，找到最优策略参数组合 |
| **策略对比** | 多策略对比工具 | 相同数据下并行运行多策略，自动排名 |
| **风险管理** | 仓位管理 + 止损止盈 | 固定比例法 / Kelly公式 / ATR自适应，自动止损止盈 |
| **技术指标** | 15+ 专业指标 | SMA/EMA/MACD/RSI/KDJ/布林带/ATR/OBV/VWAP 等 |
| **可视化** | 专业图表 | K线图、交易信号图、净值曲线、回撤图、月度收益热力图 |
| **报告生成** | 回测报告导出 | 支持文本报告 + CSV交易记录 + 净值曲线导出 |
| **CLI工具** | 8 个命令行工具 | 一键获取数据、回测、优化、对比、生成报告 |

---

## 快速开始

### 环境要求

- Python 3.10+
- pip

### 安装

```bash
# 克隆仓库
git clone https://github.com/zey413/quant-trading-system.git
cd quant-trading-system

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 安装项目及依赖
pip install -e ".[dev]"

# 验证安装
quant --help
```

### 三步上手

**第一步：获取数据**

```bash
quant fetch -s 000001 --start 2024-01-01 --end 2025-01-01
```

**第二步：运行回测**

```bash
quant backtest -st ma_crossover -s 000001 --start 2024-01-01
```

**第三步：查看图表**

```bash
quant plot -s 000001 -st ma_crossover --save chart.png
```

### Python API 快速示例

```python
from quant_trading.core import AppConfig
from quant_trading.data import DataManager
from quant_trading.backtest import BacktestEngine
from quant_trading.strategy import MACrossoverStrategy

# 加载配置 & 获取数据
config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)
df = dm.fetch_daily("000001", start_date="2024-01-01", end_date="2025-01-01")

# 创建策略 & 运行回测
strategy = MACrossoverStrategy(short_window=5, long_window=20)
engine = BacktestEngine(config)
result = engine.run(strategy, df, symbol="000001")

# 查看结果
print(f"总收益率: {result.total_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.4f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
```

---

## 策略目录

系统内置 8 大交易策略，覆盖趋势跟踪、动量突破、均值回归等主流量化思路：

| # | 策略名 | 注册名 | 原理 | 核心参数 |
|:-:|:------|:------|:-----|:--------|
| 1 | **均线交叉** | `ma_crossover` | 短期均线上穿长期均线（金叉）买入，下穿（死叉）卖出 | `short_window=5`, `long_window=20` |
| 2 | **RSI 超买超卖** | `rsi` | RSI 从超卖区上穿买入，从超买区下穿卖出 | `period=14`, `oversold=30`, `overbought=70` |
| 3 | **MACD 策略** | `macd` | MACD 柱状线由负转正（金叉）买入，由正转负（死叉）卖出 | `fast=12`, `slow=26`, `signal=9` |
| 4 | **布林带策略** | `bollinger` | 价格触及下轨后回升买入，触及上轨后回落卖出 | `window=20`, `num_std=2` |
| 5 | **Dual Thrust** | `dual_thrust` | 经典动量突破策略，基于前N日价格区间计算上下轨 | `lookback=4`, `k1=0.5`, `k2=0.5` |
| 6 | **均值回归** | `mean_reversion` | 价格偏离均线过远时（Z-Score）回归交易 | `window=20`, `entry_threshold=2.0` |
| 7 | **海龟交易** | `turtle` | 经典趋势跟踪，通道突破买入，ATR仓位管理 | `entry_period=20`, `exit_period=10`, `atr_period=20` |
| 8 | **复合策略** | `composite` | 组合多个子策略，投票决定信号（多数/全体一致/任意） | `strategies=[...]`, `voting="majority"` |

查看所有策略及参数详情：

```bash
quant strategies
```

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI 层 (Click + Rich)                     │
│       fetch │ backtest │ strategies │ plot │ optimize │ report│
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      核心层 (core)                           │
│   AppConfig │ Signal │ Order │ Position │ Portfolio          │
└───────┬──────────────┬──────────────┬───────────────────────┘
        │              │              │
┌───────┴──┐   ┌───────┴──┐   ┌──────┴──────────────────────┐
│ 数据层   │   │ 策略层   │   │       回测引擎层             │
│ AKShare  │   │ 8大策略  │   │ BacktestEngine              │
│ Tushare  │   │ 复合策略 │   │ ├── SimulatedBroker          │
│ (可扩展) │   │ (可扩展) │   │ ├── PortfolioManager         │
└──────────┘   └──────────┘   │ ├── RiskManager              │
                              │ ├── GridSearchOptimizer       │
       ┌──────────────┐       │ └── StrategyComparator        │
       │   指标层     │       └─────────────────────────────────┘
       │ 15+ 技术指标 │
       └──────────────┘       ┌──────────────────┐
                              │   可视化层       │
                              │ K线/信号/净值/回撤│
                              └──────────────────┘
```

> 详细架构文档请查看 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### 设计原则

- **可插拔架构** — 数据源、策略均通过注册表 + 抽象基类实现，新增实现零改动核心代码
- **关注点分离** — 数据获取、策略计算、订单执行、风控验证各自独立
- **配置驱动** — 所有参数通过 YAML + Pydantic 统一管理，运行时可覆盖
- **事件驱动** — 回测引擎逐日处理，完整模拟真实交易流程

---

## CLI 命令参考

系统提供 8 个命令行工具，覆盖完整量化工作流：

### `quant fetch` — 获取股票数据

```bash
quant fetch -s 000001 --start 2024-01-01 --end 2025-01-01
quant fetch -s 600519 --source akshare -n 50
```

### `quant backtest` — 运行策略回测

```bash
quant backtest -st ma_crossover -s 000001
quant backtest -st rsi -s 600519 --capital 500000
quant backtest -st macd -s 000001 --start 2023-01-01 --end 2024-12-31 --save report.png
```

### `quant strategies` — 列出所有可用策略

```bash
quant strategies
```

### `quant plot` — 生成可视化图表

```bash
quant plot -s 000001 -t candlestick
quant plot -s 600519 -st ma_crossover --save signals.png
quant plot -s 000001 -st rsi -t all
```

### `quant optimize` — 策略参数优化

```bash
quant optimize -st ma_crossover -s 000001 \
  -p '{"short_window":[3,5,10],"long_window":[15,20,30]}'

quant optimize -st rsi -s 600519 \
  -p '{"period":[10,14,20]}' -m total_return
```

### `quant report` — 生成回测报告

```bash
quant report -st ma_crossover -s 000001
quant report -st rsi -s 600519 -f csv -o ./output
```

> 所有命令均支持 `--config` / `-c` 指定配置文件，`--verbose` / `-v` 开启详细日志。

---

## 高级用法

### 参数优化 (GridSearchOptimizer)

使用网格搜索在参数空间中自动寻找最优参数组合：

```python
from quant_trading.backtest.optimizer import GridSearchOptimizer
from quant_trading.core import AppConfig

config = AppConfig.from_yaml("config/default.yaml")
optimizer = GridSearchOptimizer(config)

# 定义参数搜索空间
param_grid = {
    "short_window": [3, 5, 10],
    "long_window": [15, 20, 30],
}

# 运行优化（默认优化夏普比率）
result = optimizer.optimize(
    strategy_name="ma_crossover",
    param_grid=param_grid,
    data=data,
    symbol="000001",
    metric="sharpe_ratio",  # 可选: total_return, annualized_return, sortino_ratio 等
)

print(result.summary())
print(f"最优参数: {result.best_params}")
print(f"最优夏普: {result.best_metric_value:.4f}")

# 查看所有参数组合结果
print(result.all_results)
```

### 策略对比 (StrategyComparator)

在相同数据下运行多个策略，自动对比排名：

```python
from quant_trading.backtest.comparator import StrategyComparator
from quant_trading.core import AppConfig

config = AppConfig.from_yaml("config/default.yaml")
comparator = StrategyComparator(config)

# 添加待对比的策略
comparator.add_strategy("ma_crossover", short_window=5, long_window=20)
comparator.add_strategy("rsi", period=14)
comparator.add_strategy("macd")
comparator.add_strategy("bollinger", window=20, num_std=2)

# 运行对比
result = comparator.compare(data, symbol="000001", rank_by="sharpe_ratio")

# 查看对比报告
print(result.summary())

# 绘制净值对比图
comparator.plot_comparison(result, save_path="comparison.png")
```

### 自定义策略

继承 `Strategy` 基类，通过装饰器注册即可接入系统：

```python
from quant_trading.strategy.base import Strategy, StrategyRegistry
import pandas as pd

@StrategyRegistry.register("my_strategy")
class MyStrategy(Strategy):
    """自定义策略"""

    def __init__(self, param1: int = 10, name: str = ""):
        super().__init__(name=name)
        self.param1 = param1

    def get_params(self) -> dict:
        return {"param1": self.param1}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["signal"] = "hold"
        # 实现你的交易逻辑
        # data.loc[buy_condition, "signal"] = "buy"
        # data.loc[sell_condition, "signal"] = "sell"
        return data
```

注册后即可在 CLI 和回测引擎中使用，无需修改其他代码。

### 自定义数据源

实现 `DataSource` 接口，同样通过装饰器注册：

```python
from quant_trading.data.base import DataSource, DataSourceRegistry
import pandas as pd

@DataSourceRegistry.register("my_source")
class MyDataSource(DataSource):
    def get_name(self) -> str:
        return "my_source"

    def fetch_daily(self, symbol, start_date, end_date, adjust="qfq") -> pd.DataFrame:
        # 返回包含 date, open, high, low, close, volume 列的 DataFrame
        ...

    def fetch_stock_list(self) -> pd.DataFrame:
        # 返回包含 code, name 列的 DataFrame
        ...
```

### 回测报告生成

支持文本报告和 CSV 数据导出：

```python
from quant_trading.backtest.report import ReportGenerator

# 生成文本报告
text_report = ReportGenerator.generate_text_report(
    result=result,
    strategy_name="ma_crossover",
    symbol="000001",
    config=config,
    trades=engine.trades,
)

# 导出交易记录CSV
ReportGenerator.generate_csv_trades(engine.trades, "trades.csv")

# 导出净值曲线CSV
ReportGenerator.generate_equity_csv(result.equity_curve, "equity.csv")
```

---

## 绩效指标

回测完成后自动计算以下专业绩效指标：

| 类别 | 指标 | 说明 |
|:---:|:-----|:-----|
| **收益** | 总收益率 (`total_return`) | 回测期间总收益 |
| **收益** | 年化收益率 (`annualized_return`) | 按244交易日年化 |
| **风险** | 最大回撤 (`max_drawdown`) | 峰值到谷值的最大跌幅 |
| **风险** | 最大回撤天数 (`max_drawdown_duration`) | 最大回撤持续交易日数 |
| **风险** | 年化波动率 (`volatility`) | 日收益率标准差年化 |
| **风险调整** | 夏普比率 (`sharpe_ratio`) | (年化收益 - 无风险利率) / 波动率 |
| **风险调整** | Sortino比率 (`sortino_ratio`) | 仅考虑下行风险的夏普比率 |
| **风险调整** | Calmar比率 (`calmar_ratio`) | 年化收益率 / 最大回撤 |
| **交易** | 总交易次数 (`total_trades`) | 平仓次数 |
| **交易** | 胜率 (`win_rate`) | 盈利交易占比 |
| **交易** | 盈亏比 (`profit_loss_ratio`) | 平均盈利 / 平均亏损 |
| **交易** | 平均盈利/亏损 (`avg_win` / `avg_loss`) | 单笔平均盈亏金额 |

---

## 配置参考

编辑 `config/default.yaml` 自定义系统参数：

```yaml
# ===== 数据源配置 =====
data_source:
  default_source: "akshare"       # 数据源: akshare(免费) / tushare(需token)
  tushare_token: ""               # Tushare Pro Token

# ===== A股交易规则 =====
broker:
  commission_rate: 0.0003         # 佣金率 万三
  min_commission: 5.0             # 最低佣金 5元
  stamp_tax_rate: 0.001           # 印花税 千一(仅卖出)
  slippage: 0.001                 # 滑点 0.1%
  lot_size: 100                   # 最小交易单位 100股(1手)
  t_plus_1: true                  # T+1 交易规则

# ===== 回测参数 =====
backtest:
  initial_capital: 1000000.0      # 初始资金 100万
  start_date: "2024-01-01"        # 回测起始日期
  end_date: "2025-12-31"          # 回测结束日期
  risk_free_rate: 0.025           # 无风险利率 2.5%

# ===== 风控参数 =====
risk:
  max_position_pct: 0.3           # 单股最大仓位 30%
  max_total_position_pct: 0.8     # 总仓位上限 80%
  stop_loss_pct: 0.05             # 止损线 5%
  take_profit_pct: 0.15           # 止盈线 15%
  position_sizing: "fixed_ratio"  # 仓位算法: fixed_ratio / kelly / atr_based
```

### A股交易规则

本系统完整支持以下A股特殊规则：

| 规则 | 说明 |
|:-----|:-----|
| T+1 | 当日买入的股票次日才能卖出 |
| 最小交易单位 | 100股（1手） |
| 佣金 | 万三（0.03%），最低5元 |
| 印花税 | 千一（0.1%），仅卖出时收取 |
| 滑点模拟 | 买入价上浮、卖出价下浮 |
| 涨跌停 | 普通股票 ±10%，ST股票 ±5% |
| 交易时间 | 9:30-11:30, 13:00-15:00 |
| 年交易日 | 约244天 |

---

## 项目结构

```
quant-trading-system/
├── config/default.yaml              # 默认配置文件
├── src/quant_trading/
│   ├── core/                        # 核心数据模型和配置
│   │   ├── config.py                # AppConfig (Pydantic)
│   │   ├── models.py                # Signal, Order, Position, Portfolio
│   │   └── enums.py                 # 枚举类型
│   ├── data/                        # 数据源层（可插拔架构）
│   │   ├── base.py                  # DataSource ABC + 注册表
│   │   ├── manager.py               # DataManager 统一接口
│   │   ├── akshare_source.py        # AKShare 免费数据源
│   │   └── tushare_source.py        # Tushare Pro 数据源
│   ├── indicators/                  # 技术指标 (15+)
│   │   ├── trend.py                 # SMA, EMA, MACD
│   │   ├── oscillator.py            # RSI, KDJ
│   │   ├── volatility.py            # 布林带, ATR
│   │   └── volume.py                # OBV, VWAP
│   ├── strategy/                    # 交易策略 (8个)
│   │   ├── base.py                  # Strategy ABC + 注册表
│   │   ├── ma_crossover.py          # 均线交叉策略
│   │   ├── rsi_strategy.py          # RSI 超买超卖策略
│   │   ├── macd_strategy.py         # MACD 策略
│   │   ├── bollinger_strategy.py    # 布林带策略
│   │   ├── dual_thrust.py           # Dual Thrust 突破策略
│   │   ├── mean_reversion.py        # 均值回归策略
│   │   ├── turtle.py                # 海龟交易策略
│   │   └── composite.py             # 复合投票策略
│   ├── backtest/                    # 回测引擎
│   │   ├── engine.py                # 回测主引擎
│   │   ├── broker.py                # 模拟券商（A股规则）
│   │   ├── portfolio.py             # 组合管理
│   │   ├── metrics.py               # 绩效指标计算
│   │   ├── optimizer.py             # GridSearchOptimizer 参数优化
│   │   ├── comparator.py            # StrategyComparator 策略对比
│   │   └── report.py                # ReportGenerator 报告生成
│   ├── risk/                        # 风险管理
│   │   ├── manager.py               # 风控管理器
│   │   ├── position_sizer.py        # 仓位管理
│   │   └── stop_loss.py             # 止损止盈
│   ├── visualization/               # 可视化
│   │   └── charts.py                # 图表绘制
│   └── cli/                         # 命令行工具
│       └── main.py                  # CLI 入口 (8个命令)
├── tests/                           # 单元测试 + 集成测试
├── examples/                        # 使用示例
│   ├── quick_start.py               # 快速上手
│   ├── custom_strategy.py           # 自定义策略
│   ├── strategy_comparison.py       # 策略对比
│   ├── parameter_optimization.py    # 参数优化
│   └── risk_management.py           # 风险管理
├── docs/                            # 文档
│   └── ARCHITECTURE.md              # 架构设计文档
├── CONTRIBUTING.md                  # 贡献指南
├── CHANGELOG.md                     # 更新日志
└── pyproject.toml                   # 项目配置
```

---

## 技术栈

| 技术 | 用途 |
|:-----|:-----|
| **Python 3.10+** | 运行环境 |
| **pandas / numpy** | 数据处理和数值计算 |
| **AKShare / Tushare** | A股行情数据获取 |
| **matplotlib** | 图表绘制和可视化 |
| **Pydantic** | 配置验证和数据模型 |
| **Click + Rich** | 命令行界面和终端美化 |
| **PyArrow** | Parquet 数据缓存 |
| **pytest** | 单元测试和集成测试 |

---

## 项目路线图

- [ ] **实时行情** — 接入实时行情数据流，支持实盘信号推送
- [ ] **更多策略** — CTA因子策略、机器学习策略、多因子选股
- [ ] **Web 界面** — 基于 Streamlit/Dash 的交互式回测面板
- [ ] **数据库支持** — PostgreSQL/ClickHouse 存储历史数据和回测结果
- [ ] **多品种支持** — 期货、ETF、可转债回测
- [ ] **分布式优化** — 多进程/分布式参数搜索加速
- [ ] **涨跌停验证** — Broker 层涨跌停限制校验
- [ ] **移动止损** — 追踪止损 (Trailing Stop) 策略
- [ ] **基准对比** — 与沪深300/中证500等基准指数对比
- [ ] **Docker 部署** — 容器化部署和定时任务

---

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 运行测试（带覆盖率）
pytest tests/ -v --cov=quant_trading

# 运行特定测试
pytest tests/test_strategies.py -v
```

详细开发指南请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新历史。

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 致谢

- [AKShare](https://github.com/akfamily/akshare) — 优秀的开源金融数据接口
- [Tushare](https://tushare.pro/) — 专业的金融数据服务平台
