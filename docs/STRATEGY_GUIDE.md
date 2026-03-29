# 策略开发指南

本文档面向策略开发者，介绍如何在量化交易系统中开发、注册和测试自定义策略。

---

## 目录

- [策略体系概览](#策略体系概览)
- [StrategyBase 接口](#strategybase-接口)
- [策略注册系统](#策略注册系统)
- [开发自定义策略](#开发自定义策略)
- [复合策略开发](#复合策略开发)
- [机器学习策略](#机器学习策略)
- [策略测试](#策略测试)
- [策略信号规范](#策略信号规范)
- [技术指标使用](#技术指标使用)
- [最佳实践](#最佳实践)
- [完整示例](#完整示例)

---

## 策略体系概览

系统的策略架构基于「抽象基类 + 装饰器注册表」模式：

```
Strategy (ABC)                     StrategyRegistry (全局注册表)
    │                                     │
    ├── MACrossoverStrategy ──── @register("ma_crossover")
    ├── RSIStrategy ─────────── @register("rsi")
    ├── MACDStrategy ────────── @register("macd")
    ├── BollingerStrategy ───── @register("bollinger")
    ├── DualThrustStrategy ──── @register("dual_thrust")
    ├── MeanReversionStrategy ─ @register("mean_reversion")
    ├── TurtleStrategy ──────── @register("turtle")
    ├── CompositeStrategy ───── @register("composite")
    └── 你的自定义策略 ──────── @register("your_strategy")
```

策略一旦注册，即可在 CLI、回测引擎、参数优化器中无缝使用，无需修改任何其他代码。

---

## StrategyBase 接口

所有策略必须继承 `Strategy` 抽象基类：

```python
from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """交易策略抽象基类"""

    def __init__(self, name: str = "") -> None:
        self.name: str = name or self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Parameters
        ----------
        data : pd.DataFrame
            至少包含 date, open, high, low, close, volume 列的行情数据。

        Returns
        -------
        pd.DataFrame
            添加了 ``signal`` 列的 DataFrame。
            signal 取值: "buy" / "sell" / "hold"
        """
        ...

    def get_params(self) -> dict:
        """返回策略参数字典，用于序列化/日志/优化"""
        return {}
```

### 必须实现的方法

| 方法 | 必须 | 说明 |
|:-----|:----:|:-----|
| `generate_signals(data)` | 是 | 核心方法，接收行情数据，返回含 signal 列的 DataFrame |
| `get_params()` | 否 | 返回参数字典，用于日志记录和参数优化 |

### 两种策略接口

回测引擎同时支持两种策略接口（自动检测）：

```python
# 方式1: 批量信号生成（推荐）— 所有内置策略使用此方式
class MyStrategy(Strategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["signal"] = "hold"
        # ... 计算信号 ...
        return data

# 方式2: 逐日信号（需要组合状态信息时使用）
class MyOnBarStrategy:
    def on_bar(self, date, row, snapshot) -> Signal | None:
        """
        Args:
            date: 当前交易日
            row: 当日行情数据 (Series)
            snapshot: 组合快照 {"cash": ..., "positions": ..., "total_value": ...}
        """
        if snapshot["cash"] > 100000:
            return Signal(signal_type=SignalType.BUY, ...)
        return None
```

---

## 策略注册系统

### 注册机制

策略通过 `@StrategyRegistry.register("名称")` 装饰器注册：

```python
from quant_trading.strategy.base import Strategy, StrategyRegistry

@StrategyRegistry.register("my_strategy")
class MyStrategy(Strategy):
    ...
```

### 注册后的使用方式

```python
# 按名称获取策略实例
strategy = StrategyRegistry.get("my_strategy", param1=10, param2=0.5)

# 列出所有已注册的策略
names = StrategyRegistry.list_strategies()
print(names)  # ['ma_crossover', 'rsi', 'macd', ..., 'my_strategy']
```

### CLI 中使用

注册后可直接在命令行使用：

```bash
quant backtest -st my_strategy -s 000001
quant optimize -st my_strategy -s 000001 -p '{"param1":[5,10,15]}'
```

### 注册完整流程

1. 创建策略文件 `src/quant_trading/strategy/my_strategy.py`
2. 在文件中使用 `@StrategyRegistry.register("my_strategy")` 装饰器
3. 在 `src/quant_trading/strategy/__init__.py` 中导入新策略模块

```python
# __init__.py
from quant_trading.strategy.my_strategy import MyStrategy

__all__ = [
    ...,
    "MyStrategy",
]
```

---

## 开发自定义策略

### 完整模板

```python
"""我的自定义策略"""

from __future__ import annotations

import pandas as pd

from quant_trading.strategy.base import Strategy, StrategyRegistry
from quant_trading.indicators.trend import SMA


@StrategyRegistry.register("my_momentum")
class MomentumStrategy(Strategy):
    """动量策略

    当N日涨幅超过阈值时买入，跌幅超过阈值时卖出。

    Parameters
    ----------
    lookback : int
        动量回看周期（天数），默认 20。
    buy_threshold : float
        买入阈值（涨幅百分比），默认 0.05 (5%)。
    sell_threshold : float
        卖出阈值（跌幅百分比），默认 -0.03 (-3%)。
    """

    def __init__(
        self,
        lookback: int = 20,
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.03,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if lookback < 1:
            raise ValueError(f"lookback 必须 >= 1，收到 {lookback}")
        self.lookback = lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def get_params(self) -> dict:
        """返回策略参数"""
        return {
            "lookback": self.lookback,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        # 1. 复制数据，避免修改原始 DataFrame
        data = data.copy()

        # 2. 计算指标
        data["momentum"] = data["close"].pct_change(periods=self.lookback)

        # 3. 生成信号
        data["signal"] = "hold"

        # 买入条件: 动量 > 买入阈值
        buy_mask = data["momentum"] > self.buy_threshold
        data.loc[buy_mask, "signal"] = "buy"

        # 卖出条件: 动量 < 卖出阈值
        sell_mask = data["momentum"] < self.sell_threshold
        data.loc[sell_mask, "signal"] = "sell"

        return data
```

### 开发步骤

#### 步骤一：创建策略文件

在 `src/quant_trading/strategy/` 下新建文件：

```bash
touch src/quant_trading/strategy/my_momentum.py
```

#### 步骤二：实现策略逻辑

按上面的模板实现 `__init__`、`get_params`、`generate_signals` 三个方法。

#### 步骤三：注册到系统

在 `src/quant_trading/strategy/__init__.py` 中添加导入：

```python
from quant_trading.strategy.my_momentum import MomentumStrategy
```

#### 步骤四：编写测试

```python
# tests/test_my_momentum.py
import pandas as pd
import pytest

from quant_trading.strategy.my_momentum import MomentumStrategy


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


class TestMomentumStrategy:
    def test_generates_signals(self):
        strategy = MomentumStrategy(lookback=10, buy_threshold=0.02)
        data = make_sample_data()
        result = strategy.generate_signals(data)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})

    def test_params(self):
        strategy = MomentumStrategy(lookback=15, buy_threshold=0.05)
        params = strategy.get_params()
        assert params["lookback"] == 15
        assert params["buy_threshold"] == 0.05

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            MomentumStrategy(lookback=-1)

    def test_does_not_modify_input(self):
        strategy = MomentumStrategy()
        data = make_sample_data()
        original_cols = set(data.columns)
        strategy.generate_signals(data)
        assert set(data.columns) == original_cols  # 不应修改原始数据
```

#### 步骤五：验证

```bash
# 运行测试
pytest tests/test_my_momentum.py -v

# CLI 验证
quant strategies              # 应该看到 my_momentum
quant backtest -st my_momentum -s 000001
```

---

## 复合策略开发

复合策略通过组合多个子策略的信号来决定最终交易方向：

### 使用内置复合策略

```python
from quant_trading.strategy import StrategyRegistry

# 通过注册表获取子策略
sub_strategies = [
    StrategyRegistry.get("ma_crossover", short_window=5, long_window=20),
    StrategyRegistry.get("rsi", period=14),
    StrategyRegistry.get("macd"),
]

# 创建复合策略
composite = StrategyRegistry.get(
    "composite",
    strategies=sub_strategies,
    voting="majority",  # majority / unanimous / any
)

# 运行回测
result = engine.run(composite, data, symbol="000001")
```

### 投票机制

| 模式 | 规则 | 适用场景 |
|:-----|:-----|:---------|
| `majority` | 超过半数策略同意 | 默认，平衡性好 |
| `unanimous` | 全部策略一致 | 保守，信号少但质量高 |
| `any` | 任一策略发出 | 激进，信号多 |

### 自定义复合策略

```python
@StrategyRegistry.register("weighted_composite")
class WeightedCompositeStrategy(Strategy):
    """加权复合策略"""

    def __init__(self, strategies, weights, threshold=0.5, name=""):
        super().__init__(name=name)
        self.strategies = strategies
        self.weights = weights
        self.threshold = threshold

    def get_params(self):
        return {
            "n_strategies": len(self.strategies),
            "threshold": self.threshold,
        }

    def generate_signals(self, data):
        data = data.copy()
        buy_score = pd.Series(0.0, index=data.index)
        sell_score = pd.Series(0.0, index=data.index)

        for strategy, weight in zip(self.strategies, self.weights):
            result = strategy.generate_signals(data.copy())
            buy_score += (result["signal"] == "buy").astype(float) * weight
            sell_score += (result["signal"] == "sell").astype(float) * weight

        data["signal"] = "hold"
        data.loc[buy_score >= self.threshold, "signal"] = "buy"
        data.loc[sell_score >= self.threshold, "signal"] = "sell"

        return data
```

---

## 机器学习策略

### ML 模块概览

```
ml/
├── features.py     # FeatureEngineer - 特征工程
├── models.py       # ML模型（纯NumPy实现）
│   ├── DecisionTreeModel
│   ├── RandomForestModel
│   ├── LinearModel
│   ├── GradientBoostingModel
│   ├── LSTMModel
│   └── EnsembleModel
├── pipeline.py     # MLPipeline + 标准化器
└── evaluation.py   # ModelEvaluator
```

### 特征工程

```python
from quant_trading.ml import FeatureEngineer

fe = FeatureEngineer()

# 一键生成特征（包含技术指标、价格变化、统计特征等）
features_df = fe.create_features(data)
```

生成的特征包括：
- 技术指标类：SMA、EMA、RSI、MACD、布林带、ATR 等
- 价格变化类：日收益率、N日动量、波动率
- 统计特征类：均值、标准差、偏度、峰度

### 构建ML策略

```python
from quant_trading.ml import (
    FeatureEngineer, RandomForestModel, MLPipeline,
    StandardScaler, ModelEvaluator
)
from quant_trading.strategy.base import Strategy, StrategyRegistry
import numpy as np
import pandas as pd


@StrategyRegistry.register("ml_rf")
class RandomForestStrategy(Strategy):
    """随机森林ML策略"""

    def __init__(self, n_trees=50, max_depth=8, train_ratio=0.7, name=""):
        super().__init__(name=name)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.train_ratio = train_ratio

    def get_params(self):
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "train_ratio": self.train_ratio,
        }

    def generate_signals(self, data):
        data = data.copy()
        data["signal"] = "hold"

        # 1. 特征工程
        fe = FeatureEngineer()
        features_df = fe.create_features(data)

        # 2. 构造标签（未来N日收益率 > 0 → 1，否则 → 0）
        future_return = data["close"].pct_change(5).shift(-5)
        labels = (future_return > 0).astype(int).values

        # 3. 提取特征矩阵
        feature_cols = [c for c in features_df.columns if c not in
                       ["date", "open", "high", "low", "close", "volume", "signal"]]
        X = features_df[feature_cols].values

        # 4. 移除NaN
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(labels)
        X_valid = X[valid_mask]
        y_valid = labels[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(X_valid) < 50:
            return data

        # 5. 划分训练/测试
        split = int(len(X_valid) * self.train_ratio)
        X_train, y_train = X_valid[:split], y_valid[:split]
        X_predict = X_valid[split:]
        predict_indices = valid_indices[split:]

        # 6. 训练
        pipeline = MLPipeline(
            model=RandomForestModel(n_trees=self.n_trees, max_depth=self.max_depth),
            scaler=StandardScaler(),
        )
        pipeline.fit(X_train, y_train)

        # 7. 预测
        predictions = pipeline.predict(X_predict)

        # 8. 生成信号
        for i, idx in enumerate(predict_indices):
            if predictions[i] == 1:
                data.iloc[idx, data.columns.get_loc("signal")] = "buy"
            else:
                data.iloc[idx, data.columns.get_loc("signal")] = "sell"

        return data
```

### 可用ML模型

| 模型 | 类名 | 特点 | 核心参数 |
|:-----|:-----|:-----|:---------|
| 决策树 | `DecisionTreeModel` | 简单快速，可解释 | `max_depth`, `min_samples_split` |
| 随机森林 | `RandomForestModel` | Bagging集成，抗过拟合 | `n_trees`, `max_depth` |
| 线性模型 | `LinearModel` | 轻量，适合线性关系 | `learning_rate`, `n_iterations` |
| 梯度提升 | `GradientBoostingModel` | Boosting集成，精度高 | `n_estimators`, `learning_rate`, `max_depth` |
| LSTM | `LSTMModel` | 时序建模能力强 | `hidden_size`, `n_epochs` |
| 集成模型 | `EnsembleModel` | 多模型投票 | `models` (模型列表) |

### 模型评估

```python
from quant_trading.ml import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_true, y_pred)
print(f"准确率: {metrics['accuracy']:.2%}")
print(f"精确率: {metrics['precision']:.2%}")
print(f"召回率: {metrics['recall']:.2%}")
print(f"F1分数: {metrics['f1']:.2%}")
```

### 数据标准化

```python
from quant_trading.ml import StandardScaler, MinMaxScaler

# 标准化（零均值单位方差）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 归一化（缩放到 [0, 1]）
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

## 策略测试

### 测试要求

1. 所有策略必须有对应的单元测试
2. 使用合成数据，不依赖网络
3. 覆盖正常场景、边界条件、错误输入

### 测试模板

```python
import pandas as pd
import pytest
from quant_trading.strategy.my_strategy import MyStrategy


def make_sample_data(n_days=100):
    """合成测试数据"""
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


class TestMyStrategy:
    """自定义策略测试"""

    def test_generates_valid_signals(self):
        """信号列存在且值合法"""
        strategy = MyStrategy()
        result = strategy.generate_signals(make_sample_data())
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({"buy", "sell", "hold"})

    def test_does_not_modify_input(self):
        """不修改原始DataFrame"""
        strategy = MyStrategy()
        data = make_sample_data()
        original_cols = list(data.columns)
        strategy.generate_signals(data)
        assert list(data.columns) == original_cols

    def test_get_params(self):
        """参数序列化"""
        strategy = MyStrategy(param1=10)
        params = strategy.get_params()
        assert isinstance(params, dict)
        assert "param1" in params

    def test_empty_data(self):
        """空数据处理"""
        strategy = MyStrategy()
        empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        result = strategy.generate_signals(empty_df)
        assert "signal" in result.columns

    def test_short_data(self):
        """数据量不足时的处理"""
        strategy = MyStrategy(lookback=20)
        result = strategy.generate_signals(make_sample_data(n_days=5))
        assert "signal" in result.columns
        # 数据不足应全部为 hold
        assert all(result["signal"] == "hold")

    def test_invalid_params(self):
        """非法参数应抛出异常"""
        with pytest.raises(ValueError):
            MyStrategy(param1=-1)

    def test_registry(self):
        """确认策略已注册"""
        from quant_trading.strategy.base import StrategyRegistry
        assert "my_strategy" in StrategyRegistry.list_strategies()
```

### 运行测试

```bash
# 运行指定测试
pytest tests/test_my_strategy.py -v

# 运行所有策略测试
pytest tests/test_strategies.py -v

# 带覆盖率
pytest tests/ -v --cov=quant_trading
```

---

## 策略信号规范

### 信号格式

| 信号值 | 含义 | 触发操作 |
|:-------|:-----|:---------|
| `"buy"` | 买入信号 | 回测引擎执行买入 |
| `"sell"` | 卖出信号 | 回测引擎执行卖出（需有持仓） |
| `"hold"` | 持有/无操作 | 不执行交易 |

### 关键约束

1. **signal 列必须存在**: `generate_signals` 返回的 DataFrame 必须包含 `signal` 列
2. **值域限制**: signal 值只能是 `"buy"` / `"sell"` / `"hold"` 之一
3. **不修改原数据**: 使用 `data = data.copy()` 避免修改传入的 DataFrame
4. **不依赖网络**: 策略逻辑不应包含网络请求
5. **数据不足处理**: 当数据不足以计算指标时，应返回全 `"hold"` 而非报错

### 信号强度（可选）

如果策略产生信号强度信息，可以添加 `signal_strength` 列（0~1），风险管理器会根据强度调整仓位：

```python
data["signal"] = "buy"
data["signal_strength"] = 0.8  # 80% 信号强度
```

---

## 技术指标使用

### 使用内置指标

```python
from quant_trading.indicators.trend import SMA, EMA, MACD
from quant_trading.indicators.oscillator import RSI
from quant_trading.indicators.volatility import BollingerBands, ATR
from quant_trading.indicators.volume import OBV, VWAP
from quant_trading.indicators.momentum import ROC, WilliamsR, CCI

# 在策略中使用
def generate_signals(self, data):
    data = data.copy()

    # 添加指标
    SMA(window=20).calculate(data)     # 添加 sma_20 列
    RSI(period=14).calculate(data)     # 添加 rsi_14 列
    MACD().calculate(data)             # 添加 macd, macd_signal, macd_hist 列
    BollingerBands(window=20).calculate(data)  # 添加 bb_upper, bb_middle, bb_lower

    # 使用指标值生成信号
    data["signal"] = "hold"
    data.loc[data["rsi_14"] < 30, "signal"] = "buy"
    data.loc[data["rsi_14"] > 70, "signal"] = "sell"

    return data
```

### 一键添加所有指标

```python
from quant_trading.indicators.utils import add_all_indicators

data = add_all_indicators(data)
# 自动添加: SMA(5,10,20,60), EMA(12,26), RSI(14), MACD, BB(20),
#           ATR(14), OBV, VWAP, ROC(12), WilliamsR(14), CCI(20)

# 自定义配置
data = add_all_indicators(data, config={
    "sma": [5, 20],
    "rsi": [14, 28],
    "macd": True,
    "bollinger": False,  # 跳过布林带
})
```

### 指标相关性分析

```python
from quant_trading.indicators.utils import calculate_indicator_correlation

corr = calculate_indicator_correlation(data, ["rsi_14", "sma_20", "macd"])
print(corr)
```

### 可用指标清单

| 类别 | 指标 | 类名 | 输出列 |
|:-----|:-----|:-----|:-------|
| 趋势 | 简单移动平均 | `SMA` | `sma_{window}` |
| 趋势 | 指数移动平均 | `EMA` | `ema_{window}` |
| 趋势 | MACD | `MACD` | `macd`, `macd_signal`, `macd_hist` |
| 振荡 | RSI | `RSI` | `rsi_{period}` |
| 振荡 | KDJ | `KDJ` | `k`, `d`, `j` |
| 波动率 | 布林带 | `BollingerBands` | `bb_upper`, `bb_middle`, `bb_lower` |
| 波动率 | ATR | `ATR` | `atr_{window}` |
| 成交量 | OBV | `OBV` | `obv` |
| 成交量 | VWAP | `VWAP` | `vwap` |
| 动量 | ROC | `ROC` | `roc_{period}` |
| 动量 | Williams %R | `WilliamsR` | `williams_r_{period}` |
| 动量 | CCI | `CCI` | `cci_{period}` |

---

## 最佳实践

### 1. 始终复制数据

```python
def generate_signals(self, data):
    data = data.copy()  # 必须复制！
    ...
```

### 2. 处理数据不足的情况

```python
def generate_signals(self, data):
    data = data.copy()
    data["signal"] = "hold"

    if len(data) < self.window:
        return data  # 数据不足，全部 hold

    # ... 正常逻辑 ...
    return data
```

### 3. 参数验证

```python
def __init__(self, window: int = 20, threshold: float = 0.05):
    if window < 1:
        raise ValueError(f"window 必须 >= 1，收到 {window}")
    if not (0 < threshold < 1):
        raise ValueError(f"threshold 必须在 0~1 之间，收到 {threshold}")
```

### 4. 避免前视偏差

```python
# 错误示例 — 使用了未来数据
data["signal"] = "hold"
data.loc[data["close"].shift(-1) > data["close"], "signal"] = "buy"  # 使用了明日价格！

# 正确示例 — 只使用历史数据
data["signal"] = "hold"
data.loc[data["sma_5"] > data["sma_20"], "signal"] = "buy"
```

### 5. 信号去抖动

```python
# 避免频繁买卖
data["signal"] = "hold"
# 只在交叉发生的当天产生信号，而非持续产生
data.loc[(data["sma_5"] > data["sma_20"]) &
         (data["sma_5"].shift(1) <= data["sma_20"].shift(1)), "signal"] = "buy"
```

### 6. 使用 get_params 支持优化

```python
def get_params(self) -> dict:
    """所有可优化参数都应在此返回"""
    return {
        "window": self.window,
        "threshold": self.threshold,
    }
```

---

## 完整示例

### 示例：双均线动量策略

```python
"""双均线 + 动量过滤策略

买入条件：
1. 短期均线上穿长期均线（金叉）
2. 动量（ROC）> 0（确认趋势）

卖出条件：
1. 短期均线下穿长期均线（死叉）
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.trend import SMA
from quant_trading.indicators.momentum import ROC
from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("ma_momentum")
class MAMomentumStrategy(Strategy):
    """双均线 + 动量过滤策略"""

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        roc_period: int = 12,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if short_window >= long_window:
            raise ValueError(
                f"short_window({short_window}) 必须小于 long_window({long_window})"
            )
        self.short_window = short_window
        self.long_window = long_window
        self.roc_period = roc_period

    def get_params(self) -> dict:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "roc_period": self.roc_period,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["signal"] = "hold"

        # 数据量检查
        min_periods = max(self.long_window, self.roc_period) + 1
        if len(data) < min_periods:
            return data

        # 计算指标
        SMA(window=self.short_window).calculate(data)
        SMA(window=self.long_window).calculate(data)
        ROC(period=self.roc_period).calculate(data)

        short_col = f"sma_{self.short_window}"
        long_col = f"sma_{self.long_window}"
        roc_col = f"roc_{self.roc_period}"

        # 金叉 + 动量确认 → 买入
        golden_cross = (
            (data[short_col] > data[long_col]) &
            (data[short_col].shift(1) <= data[long_col].shift(1))
        )
        momentum_positive = data[roc_col] > 0
        data.loc[golden_cross & momentum_positive, "signal"] = "buy"

        # 死叉 → 卖出
        death_cross = (
            (data[short_col] < data[long_col]) &
            (data[short_col].shift(1) >= data[long_col].shift(1))
        )
        data.loc[death_cross, "signal"] = "sell"

        return data
```

### 示例：RSI + 布林带组合策略

```python
@StrategyRegistry.register("rsi_bollinger")
class RSIBollingerStrategy(Strategy):
    """RSI + 布林带组合策略

    买入: RSI超卖 + 价格触及布林带下轨
    卖出: RSI超买 + 价格触及布林带上轨
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
        from quant_trading.indicators.oscillator import RSI
        from quant_trading.indicators.volatility import BollingerBands

        data = data.copy()
        data["signal"] = "hold"

        RSI(period=self.rsi_period).calculate(data)
        BollingerBands(window=self.bb_window, num_std=self.bb_std).calculate(data)

        rsi_col = f"rsi_{self.rsi_period}"

        # 买入：RSI 超卖 + 价格触及下轨
        buy_mask = (
            (data[rsi_col] < self.rsi_oversold) &
            (data["close"] <= data["bb_lower"])
        )
        data.loc[buy_mask, "signal"] = "buy"

        # 卖出：RSI 超买 + 价格触及上轨
        sell_mask = (
            (data[rsi_col] > self.rsi_overbought) &
            (data["close"] >= data["bb_upper"])
        )
        data.loc[sell_mask, "signal"] = "sell"

        return data
```
