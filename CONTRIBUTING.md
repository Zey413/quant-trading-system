# 贡献指南

感谢您对本项目的关注！本文档介绍如何参与 A 股量化交易系统的开发。

## 目录

- [开发环境搭建](#开发环境搭建)
- [代码规范](#代码规范)
- [如何添加新策略](#如何添加新策略)
- [如何添加新数据源](#如何添加新数据源)
- [如何添加新指标](#如何添加新指标)
- [提交 PR 的流程](#提交-pr-的流程)
- [测试要求](#测试要求)

---

## 开发环境搭建

### 前置条件

- Python 3.10+
- Git

### 安装步骤

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/<your-username>/quant-trading-system.git
cd quant-trading-system

# 2. 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# 3. 安装项目及开发依赖
pip install -e ".[dev]"

# 4. 验证安装
python -m pytest tests/ -v
quant --help
```

### 项目结构

```
src/quant_trading/
├── core/           # 核心数据模型、枚举、配置
├── data/           # 数据源层（可插拔架构）
├── indicators/     # 技术指标库
├── strategy/       # 交易策略
├── backtest/       # 回测引擎
├── risk/           # 风险管理
├── visualization/  # 图表绘制
└── cli/            # 命令行工具
```

---

## 代码规范

### 语言与编码

- 代码注释、文档字符串使用**中文**（面向国内 A 股用户）
- 变量名、函数名、类名使用**英文**
- 文件编码统一使用 **UTF-8**

### 命名规范

| 类型 | 风格 | 示例 |
|------|------|------|
| 模块文件 | snake_case | `ma_crossover.py` |
| 类名 | PascalCase | `MACrossoverStrategy` |
| 函数/方法 | snake_case | `generate_signals()` |
| 常量 | UPPER_SNAKE_CASE | `TRADING_DAYS_PER_YEAR` |
| 私有方法 | 前缀下划线 | `_calc_buy_quantity()` |

### 类型注解

所有公开函数和方法必须添加类型注解：

```python
def calculate_quantity(
    self,
    signal: Signal,
    portfolio: Portfolio,
    method: str | None = None,
) -> int:
    """计算买入数量"""
    ...
```

### 文档字符串

使用 Google/NumPy 风格（项目中以 NumPy 风格为主）：

```python
def fetch_daily(
    self,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """获取日线行情数据

    Parameters
    ----------
    symbol : str
        股票代码，如 "000001"
    start_date : str
        开始日期，"YYYY-MM-DD" 格式
    end_date : str
        结束日期，"YYYY-MM-DD" 格式

    Returns
    -------
    pd.DataFrame
        包含 date, open, high, low, close, volume 列的 DataFrame
    """
```

### 导入顺序

```python
# 1. 标准库
from __future__ import annotations
import logging
from datetime import date

# 2. 第三方库
import numpy as np
import pandas as pd

# 3. 本项目模块
from quant_trading.core.models import Signal
from quant_trading.strategy.base import Strategy
```

---

## 如何添加新策略

### 步骤一：创建策略文件

在 `src/quant_trading/strategy/` 下新建文件，例如 `my_strategy.py`：

```python
"""我的自定义策略"""

from __future__ import annotations

import pandas as pd

from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("my_strategy")
class MyStrategy(Strategy):
    """自定义策略描述

    Parameters
    ----------
    param1 : int
        参数1说明
    param2 : float
        参数2说明
    """

    def __init__(
        self,
        param1: int = 10,
        param2: float = 0.5,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.param1 = param1
        self.param2 = param2

    def get_params(self) -> dict:
        return {"param1": self.param1, "param2": self.param2}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """实现信号生成逻辑

        必须在 data 中添加 'signal' 列，值为 "buy" / "sell" / "hold"。
        """
        data = data.copy()
        data["signal"] = "hold"

        # --- 你的交易逻辑 ---
        # 可使用 indicators 模块计算技术指标
        # data.loc[buy_condition, "signal"] = "buy"
        # data.loc[sell_condition, "signal"] = "sell"

        return data
```

### 步骤二：注册策略

在 `src/quant_trading/strategy/__init__.py` 中导入新策略：

```python
from quant_trading.strategy.my_strategy import MyStrategy

# 并添加到 __all__
__all__ = [
    ...,
    "MyStrategy",
]
```

### 步骤三：编写测试

在 `tests/test_strategies.py` 中添加对应测试：

```python
class TestMyStrategy:
    def test_generates_signals(self):
        strategy = MyStrategy(param1=10, param2=0.5)
        data = make_sample_data()  # 使用合成数据
        result = strategy.generate_signals(data)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            MyStrategy(param1=-1)
```

### 关键约束

- `generate_signals()` 必须返回包含 `signal` 列的 DataFrame
- signal 值只能是 `"buy"` / `"sell"` / `"hold"` 之一
- 策略不应修改传入的原始 DataFrame（使用 `.copy()`）
- 策略不应依赖网络调用

---

## 如何添加新数据源

### 步骤一：创建数据源文件

在 `src/quant_trading/data/` 下新建文件，例如 `my_source.py`：

```python
"""自定义数据源"""

from __future__ import annotations

import pandas as pd

from quant_trading.data.base import DataSource, DataSourceRegistry


@DataSourceRegistry.register("my_source")
class MyDataSource(DataSource):
    """自定义数据源说明"""

    def get_name(self) -> str:
        return "my_source"

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线数据

        返回的 DataFrame 必须包含标准列：
        date, open, high, low, close, volume
        """
        # --- 你的数据获取逻辑 ---
        df = pd.DataFrame(...)
        return df

    def fetch_stock_list(self) -> pd.DataFrame:
        """获取股票列表

        返回包含 code, name 列的 DataFrame。
        """
        return pd.DataFrame(...)
```

### 步骤二：注册数据源

在 `src/quant_trading/data/__init__.py` 中导入新数据源：

```python
import quant_trading.data.my_source as _my_source  # noqa: F401
```

### 关键约束

- 返回的 DataFrame 必须包含标准列：`date`, `open`, `high`, `low`, `close`, `volume`
- `date` 列可以是字符串或 datetime，DataManager 会自动转换
- 数据必须按日期升序排列

---

## 如何添加新指标

### 步骤一：选择指标分类

根据指标类型放入对应文件：

| 类型 | 文件 | 示例 |
|------|------|------|
| 趋势 | `indicators/trend.py` | SMA, EMA, MACD |
| 振荡 | `indicators/oscillator.py` | RSI, KDJ |
| 波动率 | `indicators/volatility.py` | 布林带, ATR |
| 成交量 | `indicators/volume.py` | OBV, VWAP |

### 步骤二：实现指标类

```python
from quant_trading.indicators.base import Indicator

class MyIndicator(Indicator):
    """我的自定义指标

    Parameters
    ----------
    window : int
        计算窗口
    """

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"my_indicator_{self.window}"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]  # 声明依赖的列

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)  # 校验输入

        # --- 计算逻辑 ---
        data[f"my_ind_{self.window}"] = data["close"].rolling(
            window=self.window
        ).mean()

        return data
```

### 关键约束

- 必须继承 `Indicator` 基类
- 必须实现 `name`、`required_columns`、`calculate` 三个接口
- `calculate` 方法在原 DataFrame 上添加列并返回
- 调用 `self._validate(data)` 校验输入列
- 使用 `min_periods` 参数避免 NaN 扩散

---

## 提交 PR 的流程

### 1. 创建分支

```bash
git checkout -b feature/your-feature-name
```

分支命名规范：
- `feature/xxx` — 新功能
- `fix/xxx` — 修复 Bug
- `refactor/xxx` — 重构
- `docs/xxx` — 文档更新
- `test/xxx` — 测试补充

### 2. 开发与提交

```bash
# 开发代码...
# 编写/补充测试...

# 运行测试确保通过
python -m pytest tests/ -v

# 提交代码
git add <files>
git commit -m "feat: 添加XXX策略"
```

Commit message 格式：
- `feat: xxx` — 新功能
- `fix: xxx` — Bug 修复
- `refactor: xxx` — 重构
- `docs: xxx` — 文档
- `test: xxx` — 测试
- `chore: xxx` — 构建/工具

### 3. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

在 GitHub 上创建 Pull Request，描述中说明：
- 改动内容和目的
- 测试情况
- 是否有破坏性变更

### 4. Code Review

- 至少需要 1 位 Reviewer 审核
- 解决所有评审意见后方可合并
- 确保 CI 测试全部通过

---

## 测试要求

### 基本要求

- 所有新增代码必须包含对应的单元测试
- 测试通过率 100%
- 测试必须可离线运行（不依赖外部 API）

### 测试文件位置

```
tests/
├── test_backtest.py        # 回测模块测试
├── test_strategies.py      # 策略测试
├── test_indicators.py      # 指标测试
├── test_risk.py            # 风控测试
├── test_data_sources.py    # 数据源测试
└── test_integration.py     # 集成测试
```

### 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_strategies.py -v

# 运行带覆盖率报告
python -m pytest tests/ -v --cov=quant_trading

# 运行指定测试类
python -m pytest tests/test_integration.py::TestMACrossoverIntegration -v
```

### 测试编写规范

```python
import pytest
import pandas as pd

class TestMyFeature:
    """功能名称测试"""

    def test_normal_case(self):
        """正常情况下的预期行为"""
        ...

    def test_edge_case(self):
        """边界条件处理"""
        ...

    def test_error_handling(self):
        """错误输入应抛出合适的异常"""
        with pytest.raises(ValueError, match="错误描述"):
            ...
```

### 使用合成数据

所有测试必须使用合成数据，禁止调用真实 API：

```python
def make_sample_data(n_days=50):
    """创建合成测试数据"""
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    prices = [10.0 * (1.002 ** i) for i in range(n_days)]
    return pd.DataFrame({
        "date": dates,
        "open": [p * 0.998 for p in prices],
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * n_days,
    })
```

---

## 问题反馈

如果您在贡献过程中遇到问题，欢迎通过以下渠道反馈：

- 提交 [GitHub Issue](https://github.com/zey413/quant-trading-system/issues)
- 在 PR 中留言讨论
