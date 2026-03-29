# 量化交易系统 (Quant Trading System)

> 面向中国A股市场的量化交易系统，支持可插拔数据源、多策略回测、风险管理

## 功能特性

- **可插拔数据源**: 支持 AKShare (免费) 和 Tushare Pro，可轻松扩展新数据源
- **多策略引擎**: 内置均线交叉、RSI、MACD、布林带等策略，支持自定义策略和复合策略
- **专业回测**: 事件驱动回测引擎，完整支持A股交易规则 (T+1、印花税、最小交易单位等)
- **技术指标库**: SMA/EMA、RSI/KDJ、MACD、布林带、ATR、OBV/VWAP 等 15+ 指标
- **风险管理**: 仓位管理、止损止盈、持仓比例限制
- **绩效分析**: 夏普比率、最大回撤、年化收益率、Sortino比率等专业指标
- **可视化**: K线图、交易信号图、净值曲线、回撤图
- **命令行工具**: 一键获取数据、运行回测、查看结果

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/zey413/quant-trading-system.git
cd quant-trading-system

# 安装依赖 (推荐使用虚拟环境)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 基本用法

#### 1. 获取股票数据

```python
from quant_trading.data import DataManager
from quant_trading.core import AppConfig

config = AppConfig.from_yaml("config/default.yaml")
dm = DataManager(config)

# 获取平安银行日线数据
df = dm.fetch_daily("000001", start_date="2024-01-01", end_date="2025-01-01")
print(df.head())
```

#### 2. 运行回测

```python
from quant_trading.backtest import BacktestEngine
from quant_trading.strategy import MACrossoverStrategy

# 创建策略
strategy = MACrossoverStrategy(short_window=5, long_window=20)

# 运行回测
engine = BacktestEngine(config)
result = engine.run(strategy, symbols=["000001", "600519"])

# 查看结果
print(result.summary())
```

#### 3. 命令行工具

```bash
# 获取数据
quant fetch --symbol 000001 --start 2024-01-01 --end 2025-01-01

# 运行回测
quant backtest --strategy ma_crossover --symbol 000001 --start 2024-01-01

# 查看可用策略
quant strategies

# 生成图表
quant plot --symbol 000001 --strategy ma_crossover
```

## 项目结构

```
quant-trading-system/
├── config/default.yaml          # 默认配置
├── src/quant_trading/
│   ├── core/                    # 核心数据模型和配置
│   │   ├── config.py            # 应用配置 (Pydantic)
│   │   ├── models.py            # 数据模型 (Signal, Order, Position, Portfolio)
│   │   └── enums.py             # 枚举类型
│   ├── data/                    # 数据源层 (可插拔架构)
│   │   ├── base.py              # DataSource ABC + 注册表
│   │   ├── manager.py           # DataManager 统一接口
│   │   ├── akshare_source.py    # AKShare 数据源
│   │   └── tushare_source.py    # Tushare Pro 数据源
│   ├── indicators/              # 技术指标
│   │   ├── trend.py             # SMA, EMA, MACD
│   │   ├── oscillator.py        # RSI, KDJ
│   │   ├── volatility.py        # 布林带, ATR
│   │   └── volume.py            # OBV, VWAP
│   ├── strategy/                # 交易策略
│   │   ├── base.py              # Strategy ABC + 注册表
│   │   ├── ma_crossover.py      # 均线交叉策略
│   │   ├── rsi_strategy.py      # RSI 策略
│   │   ├── macd_strategy.py     # MACD 策略
│   │   ├── bollinger_strategy.py # 布林带策略
│   │   └── composite.py         # 复合策略
│   ├── backtest/                # 回测引擎
│   │   ├── engine.py            # 回测主引擎
│   │   ├── broker.py            # 模拟券商 (A股规则)
│   │   ├── portfolio.py         # 组合管理
│   │   └── metrics.py           # 绩效指标
│   ├── risk/                    # 风险管理
│   │   ├── manager.py           # 风控管理器
│   │   ├── position_sizer.py    # 仓位管理
│   │   └── stop_loss.py         # 止损止盈
│   ├── visualization/           # 可视化
│   │   └── charts.py            # 图表绘制
│   └── cli/                     # 命令行工具
│       └── main.py              # CLI 入口
├── tests/                       # 单元测试
├── examples/                    # 使用示例
└── pyproject.toml               # 项目配置
```

## 配置说明

编辑 `config/default.yaml` 来自定义系统参数：

```yaml
data_source:
  default_source: "akshare"     # 数据源: akshare(免费) / tushare(需token)
  tushare_token: ""             # Tushare Pro Token

broker:
  commission_rate: 0.0003       # 佣金率 万三
  stamp_tax_rate: 0.001         # 印花税 千一(仅卖出)
  lot_size: 100                 # 最小交易单位 100股
  t_plus_1: true                # T+1规则

backtest:
  initial_capital: 1000000.0    # 初始资金 100万
  risk_free_rate: 0.025         # 无风险利率

risk:
  max_position_pct: 0.3         # 单股最大仓位 30%
  stop_loss_pct: 0.05           # 止损线 5%
```

## 自定义策略

继承 `Strategy` 基类即可创建自定义策略：

```python
from quant_trading.strategy.base import Strategy, StrategyRegistry
import pandas as pd

@StrategyRegistry.register("my_strategy")
class MyStrategy(Strategy):
    """自定义策略示例"""

    def __init__(self, param1: int = 10):
        super().__init__(name="my_strategy")
        self.param1 = param1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = data.copy()
        # 实现你的交易逻辑
        data["signal"] = "hold"
        # ... 你的信号逻辑
        return data
```

## 自定义数据源

实现 `DataSource` 接口即可接入新数据源：

```python
from quant_trading.data.base import DataSource, DataSourceRegistry
import pandas as pd

@DataSourceRegistry.register("my_source")
class MyDataSource(DataSource):
    def get_name(self) -> str:
        return "my_source"

    def fetch_daily(self, symbol, start_date, end_date, adjust="qfq") -> pd.DataFrame:
        # 实现数据获取逻辑
        # 返回包含 date, open, high, low, close, volume 列的DataFrame
        ...

    def fetch_stock_list(self) -> pd.DataFrame:
        # 返回股票列表
        ...
```

## A股交易规则

本系统完整支持以下A股特殊规则：

| 规则 | 说明 |
|------|------|
| T+1 | 当日买入的股票次日才能卖出 |
| 最小交易单位 | 100股(1手) |
| 佣金 | 万三(0.03%)，最低5元 |
| 印花税 | 千一(0.1%)，仅卖出时收取 |
| 涨跌停 | 普通股票 ±10%，ST股票 ±5% |
| 交易时间 | 9:30-11:30, 13:00-15:00 |
| 年交易日 | 约244天 |

## 技术栈

- **Python** 3.10+
- **pandas** / **numpy** - 数据处理和数值计算
- **AKShare** / **Tushare** - A股数据获取
- **matplotlib** - 图表绘制
- **Pydantic** - 配置验证
- **Click** / **Rich** - 命令行界面
- **pytest** - 单元测试

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 运行测试(带覆盖率)
pytest tests/ -v --cov=quant_trading
```

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 致谢

- [AKShare](https://github.com/akfamily/akshare) - 优秀的开源金融数据接口
- [Tushare](https://tushare.pro/) - 专业的金融数据服务平台
