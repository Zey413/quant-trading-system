# 更新日志

本文件记录量化交易系统的所有重要变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [0.5.0] - 2026-03-29

### 新增

#### 追踪止损 (`risk/stop_loss.py`)
- `TrailingStopManager` 追踪止损管理器
- `TrailingStopMode.PERCENT` 固定百分比追踪：止损价 = 最高价 × (1 - trail_pct)
- `TrailingStopMode.ATR` ATR倍数追踪：止损价 = 最高价 - multiplier × ATR
- `TrailingStopMode.STEP` 阶梯式追踪：盈利越多止损越紧
- 止损价只上移不下移，支持多股票独立追踪

#### 涨跌停验证 (`backtest/broker.py`)
- `PriceLimitChecker` 涨跌停限制校验器
- 支持主板(±10%)、ST(±5%)、科创板(±20%)、创业板(±20%)、北交所(±30%)
- `SimulatedBroker` 集成涨跌停校验，涨停不可买入，跌停不可卖出
- 可通过 `enable_price_limit` 开关控制

#### 基准对比 (`backtest/benchmark.py`)
- `BenchmarkManager` 基准管理器
- 支持沪深300、中证500、上证50等主要指数
- Alpha、Beta、信息比率(IR)、超额收益计算
- 基准相关性分析

#### 因子分析框架 (`factor/`)
- `Factor` 因子基类 + `FactorRegistry` 注册表
- `MomentumFactor` 动量因子（短期/中期/长期动量）
- `ValueFactor` 价值因子（PE、PB、PS）
- `QualityFactor` 质量因子（ROE、ROA）
- `FactorAnalyzer` 因子分析器（IC、IR、分层回测）

#### 实时行情增强 (`data/realtime.py`)
- 多股票并发实时行情获取
- 订阅/回调模式（Observer Pattern）
- asyncio 异步支持
- 实时行情缓存

#### 通知推送系统 (`notification/`)
- `Notifier` 通知基类
- `EmailNotifier` SMTP邮件推送
- `WebhookNotifier` 钉钉/企业微信/Slack Webhook推送
- `NotificationManager` 统一通知管理

#### 任务调度器 (`scheduler/`)
- `TaskScheduler` 任务调度器
- 定时获取行情、定时运行策略信号
- 盘前/盘后自动任务

#### Web Dashboard 完善 (`web/`)
- 6个页面全面重写和增强
- Dashboard 首页：概览统计、策略表现、市场快照
- 回测页面：交互式策略选择、参数配置、结果展示
- 市场数据页面：K线图、指标叠加、多股票对比
- 组合管理页面：组合构建、权重调整、绩效分析
- 风控页面：仓位管理、止损设置、风控规则
- 设置页面：配置管理（数据源、券商参数）
- `charts.py` 增强：Plotly交互式图表
- `components.py` 增强：可复用Streamlit组件

#### 测试新增
- `tests/test_trailing_stop.py` 追踪止损与涨跌停测试 (29个)
- `tests/test_benchmark.py` 基准对比测试
- `tests/test_factor.py` 因子分析测试
- `tests/test_notification.py` 通知模块测试
- `tests/test_scheduler.py` 调度器测试

---

## [0.4.0] - 2026-03-29

### 新增

#### 机器学习策略模块 (`ml/`)
- `FeatureEngineer` 特征工程器，一键生成技术指标特征
- `DecisionTreeModel` 决策树分类器（纯NumPy实现）
- `RandomForestModel` 随机森林，Bagging集成多棵决策树
- `LinearModel` 线性模型（逻辑回归/线性回归）
- `GradientBoostingModel` 梯度提升决策树
- `LSTMModel` 长短期记忆网络
- `EnsembleModel` 集成模型，组合多种ML模型投票
- `MLPipeline` 训练流水线，含StandardScaler/MinMaxScaler
- `ModelEvaluator` 模型评估器（准确率、精确率、召回率、F1）

#### 交易引擎模块 (`trading/`)
- `ExecutionEngine` 信号执行与撮合引擎（模拟/实盘双模式）
- `SimulatedExecutor` 模拟交易执行器
- `LiveExecutor` 实盘交易执行器接口
- `OrderManager` 订单生命周期管理（创建、验证、执行、撤销）
- `PositionTracker` 持仓跟踪与实时盈亏计算
- `RiskMonitor` 实时风控监控器
- `RiskRule` 风控规则抽象基类
- `MaxPositionRule` / `MaxDrawdownRule` / `DailyLossLimitRule` / `ConcentrationRule` 四大风控规则
- `TradeLogger` 交易日志与统计

#### 回测引擎增强
- `WalkForwardAnalyzer` 滚动前进分析，训练/测试窗口自动切分，防止过拟合
- `PortfolioBacktester` 多股票组合回测，支持等权/自定义权重、相关性分析
- `SignalAnalyzer` 信号质量分析器，评估信号预测能力、连续盈亏统计

#### 技术指标增强 (`indicators/`)
- `ROC` 变动率指标
- `WilliamsR` 威廉指标 (%R)
- `CCI` 顺势指标
- `add_all_indicators()` 一键添加所有常用指标
- `calculate_indicator_correlation()` 指标相关性分析

#### 数据层增强
- `DataExporter` 统一数据导出工具，支持 CSV / Excel / JSON 格式

#### 部署支持
- `Dockerfile` Docker 镜像构建
- `docker-compose.yml` Docker Compose 编排

#### 文档体系
- `docs/USER_GUIDE.md` 用户手册（安装配置、回测教程、FAQ）
- `docs/STRATEGY_GUIDE.md` 策略开发指南
- `docs/API_REFERENCE.md` API 参考文档
- `docs/DEPLOYMENT.md` 部署文档
- `docs/EXAMPLES.md` 使用示例集（10个完整示例）
- 全面更新 README.md，新增架构图、ML策略、交易引擎等文档

---

## [0.3.0] - 2026-03-29

### 新增
- Dual Thrust 突破策略
- 均值回归策略
- 海龟交易策略
- `ReportGenerator` 回测报告生成器
- `StrategyComparator` 策略对比工具
- `DataExporter` 数据导出
- CLI `optimize` / `report` 命令
- 3 个新示例

---

## [0.2.0] - 2026-03-29

### 新增
- `GridSearchOptimizer` 参数优化器
- `logging_config` 日志系统
- `CONTRIBUTING.md` 贡献指南
- 完整集成测试

### 修复
- FutureWarning 兼容性问题
- Protocol 兼容性

### 增强
- 配置验证增强（Pydantic field_validator + model_validator）

---

## [0.1.0] - 2026-03-29

### 初始发布
- 完整的量化交易框架
- 5 个内置策略（均线交叉、RSI、MACD、布林带、复合）
- AKShare + Tushare 可插拔数据源
- 事件驱动回测引擎（支持A股T+1、佣金、印花税、滑点）
- 风险管理（固定比例、Kelly公式、ATR自适应仓位管理）
- 止损止盈系统
- 15+ 技术指标
- CLI 工具（6个命令）
- 可视化（K线图、信号图、净值曲线、回撤图、月度热力图）
