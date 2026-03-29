"""回测引擎 - 核心回测流程控制

组织整个回测的流程：
1. 按交易日遍历数据
2. 管理T+1规则
3. 更新价格
4. 获取策略信号
5. 风险检查
6. 订单验证与执行
7. 组合更新
8. 记录净值曲线
9. 计算绩效指标
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Protocol

import pandas as pd

from quant_trading.backtest.broker import SimulatedBroker
from quant_trading.backtest.metrics import PerformanceMetrics, PerformanceResult
from quant_trading.backtest.portfolio import PortfolioManager
from quant_trading.core.config import AppConfig
from quant_trading.core.enums import OrderSide, OrderStatus, SignalType
from quant_trading.core.models import Order, Signal, TradeRecord

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    """策略协议 - 任何策略需要实现此接口"""

    def on_bar(self, date: date, row: pd.Series, portfolio_snapshot: dict) -> Signal | None:
        """根据当日行情数据生成交易信号

        Args:
            date: 当前交易日
            row: 当日行情数据（包含open, high, low, close, volume等）
            portfolio_snapshot: 当前组合快照

        Returns:
            交易信号或None（无信号）
        """
        ...


class BacktestEngine:
    """回测引擎

    核心回测流程控制器，按交易日迭代执行策略并模拟交易。

    支持单标的回测（run）和多标的回测（run_multi）。

    Attributes:
        config: 全局配置
        broker: 模拟券商
        portfolio_manager: 组合管理器
        risk_manager: 风险管理器（可选）
        trades: 成交记录列表
        equity_curve_data: 净值曲线数据 {date: total_value}
    """

    def __init__(self, config: AppConfig) -> None:
        """初始化回测引擎

        Args:
            config: 全局配置（含券商、回测、风险配置）
        """
        self.config = config
        self.broker = SimulatedBroker(config.broker)
        self.portfolio_manager = PortfolioManager(config.backtest.initial_capital)
        self.risk_manager = None  # 延迟导入，避免循环依赖
        self.trades: list[TradeRecord] = []
        self.equity_curve_data: dict[date, float] = {}

    def _init_risk_manager(self) -> None:
        """尝试初始化风险管理器"""
        if self.risk_manager is not None:
            return
        try:
            from quant_trading.risk.manager import RiskManager

            self.risk_manager = RiskManager(self.config.risk)
        except ImportError:
            logger.info("风险管理模块未安装，跳过风险检查")
            self.risk_manager = None

    def _reset(self) -> None:
        """重置引擎状态，用于新一轮回测"""
        self.portfolio_manager = PortfolioManager(self.config.backtest.initial_capital)
        self.trades = []
        self.equity_curve_data = {}

    def run(
        self,
        strategy,
        data: pd.DataFrame,
        symbol: str = "unknown",
    ) -> PerformanceResult:
        """运行单标的回测

        支持两种策略接口:
        1. on_bar(date, row, snapshot) -> Signal (逐日方式)
        2. generate_signals(data) -> DataFrame with 'signal' column (批量方式)

        Args:
            strategy: 策略实例
            data: 行情数据DataFrame
            symbol: 股票代码

        Returns:
            PerformanceResult: 回测绩效结果
        """
        self._reset()
        self._init_risk_manager()

        # 如果策略有 generate_signals 方法，先预计算信号
        if hasattr(strategy, "generate_signals") and not hasattr(strategy, "on_bar"):
            data = strategy.generate_signals(data.copy())

        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            logger.warning("数据为空，无法运行回测")
            return PerformanceMetrics._empty_result()

        # 按日期筛选
        start_date = pd.Timestamp(self.config.backtest.start_date).date()
        end_date = pd.Timestamp(self.config.backtest.end_date).date()

        logger.info(
            "开始回测: %s, 区间 %s ~ %s, 初始资金 %.2f",
            symbol,
            start_date,
            end_date,
            self.config.backtest.initial_capital,
        )

        for idx, row in df.iterrows():
            current_date = pd.Timestamp(idx).date()

            if current_date < start_date or current_date > end_date:
                continue

            self._process_day(
                strategy=strategy,
                current_date=current_date,
                symbol=symbol,
                row=row,
                prices={symbol: row["close"]},
            )

        # 计算绩效
        result = self._calc_result()

        logger.info(
            "回测完成: 总收益率 %.2f%%, 最大回撤 %.2f%%, 夏普比率 %.4f",
            result.total_return * 100,
            result.max_drawdown * 100,
            result.sharpe_ratio,
        )

        return result

    def run_multi(
        self,
        strategy: Strategy,
        data_dict: dict[str, pd.DataFrame],
    ) -> PerformanceResult:
        """运行多标的回测

        每个交易日遍历所有标的，获取信号并执行。

        Args:
            strategy: 策略实例（on_bar接收每个标的的数据行）
            data_dict: {symbol: DataFrame} 多标的行情数据

        Returns:
            PerformanceResult: 回测绩效结果
        """
        self._reset()
        self._init_risk_manager()

        if not data_dict:
            logger.warning("数据字典为空，无法运行回测")
            return PerformanceMetrics._empty_result()

        # 准备所有数据
        prepared: dict[str, pd.DataFrame] = {}
        all_dates: set[date] = set()

        for symbol, df in data_dict.items():
            prep = self._prepare_data(df)
            if not prep.empty:
                prepared[symbol] = prep
                for idx in prep.index:
                    d = pd.Timestamp(idx).date()
                    all_dates.add(d)

        if not all_dates:
            return PerformanceMetrics._empty_result()

        # 按日期排序
        start_date = pd.Timestamp(self.config.backtest.start_date).date()
        end_date = pd.Timestamp(self.config.backtest.end_date).date()
        sorted_dates = sorted(d for d in all_dates if start_date <= d <= end_date)

        logger.info(
            "开始多标的回测: %d只股票, 区间 %s ~ %s",
            len(prepared),
            start_date,
            end_date,
        )

        for current_date in sorted_dates:
            # 1. 新交易日处理
            self.portfolio_manager.new_trading_day()

            # 2. 收集当日所有标的的价格
            prices: dict[str, float] = {}
            for symbol, df in prepared.items():
                if current_date in df.index:
                    prices[symbol] = df.loc[current_date, "close"]
                elif pd.Timestamp(current_date) in df.index:
                    prices[symbol] = df.loc[pd.Timestamp(current_date), "close"]

            # 3. 更新持仓价格
            self.portfolio_manager.update_prices(prices)

            # 4. 检查风险止损/止盈
            self._check_risk_exits(current_date, prices)

            # 5. 遍历每个标的获取信号
            for symbol, df in prepared.items():
                ts_date = pd.Timestamp(current_date)
                if current_date in df.index:
                    row = df.loc[current_date]
                elif ts_date in df.index:
                    row = df.loc[ts_date]
                else:
                    continue

                self._process_signal(
                    strategy=strategy,
                    current_date=current_date,
                    symbol=symbol,
                    row=row,
                    current_price=row["close"],
                )

            # 6. 记录净值
            self.equity_curve_data[current_date] = (
                self.portfolio_manager.portfolio.total_value
            )

        return self._calc_result()

    def _process_day(
        self,
        strategy: Strategy,
        current_date: date,
        symbol: str,
        row: pd.Series,
        prices: dict[str, float],
    ) -> None:
        """处理单个交易日

        Args:
            strategy: 策略实例
            current_date: 当前日期
            symbol: 股票代码
            row: 当日行情数据
            prices: 当日价格字典
        """
        # 1. 新交易日处理（T+1规则）
        self.portfolio_manager.new_trading_day()

        # 2. 更新持仓价格
        self.portfolio_manager.update_prices(prices)

        # 3. 检查风险止损/止盈
        self._check_risk_exits(current_date, prices)

        # 4. 获取策略信号并处理
        self._process_signal(
            strategy=strategy,
            current_date=current_date,
            symbol=symbol,
            row=row,
            current_price=row["close"],
        )

        # 5. 记录当日净值
        self.equity_curve_data[current_date] = (
            self.portfolio_manager.portfolio.total_value
        )

    def _process_signal(
        self,
        strategy,
        current_date: date,
        symbol: str,
        row: pd.Series,
        current_price: float,
    ) -> None:
        """处理策略信号

        兼容两种策略接口:
        1. on_bar(date, row, snapshot) -> Signal | None (Protocol方式)
        2. 预计算的 signal 列 (generate_signals方式)

        获取信号 -> 风险验证 -> 创建订单 -> 验证订单 -> 执行订单 -> 更新组合
        """
        signal = None

        # 方式1: 如果策略有 on_bar 方法
        if hasattr(strategy, "on_bar"):
            snapshot = self.portfolio_manager.get_snapshot()
            signal = strategy.on_bar(current_date, row, snapshot)
        # 方式2: 如果行数据中有预计算的 signal 列
        elif "signal" in row.index:
            sig_type = row["signal"]
            if sig_type in ("buy", "sell"):
                signal = Signal(
                    date=current_date,
                    symbol=symbol,
                    signal_type=sig_type,
                    price=current_price,
                )

        if signal is None or signal.signal_type == "hold":
            return

        # 风险验证
        if self.risk_manager is not None:
            valid, reason = self.risk_manager.validate_signal(
                signal, self.portfolio_manager.portfolio
            )
            if not valid:
                logger.debug(
                    "信号被风险管理拒绝: %s %s, 原因: %s",
                    signal.signal_type,
                    symbol,
                    reason,
                )
                return

        # 确定下单数量
        if signal.is_buy():
            quantity = self._calc_buy_quantity(signal, current_price)
            if quantity <= 0:
                return
            self._execute_buy(symbol, quantity, current_price, current_date)
        elif signal.is_sell():
            self._execute_sell(symbol, current_price, current_date)

    def _calc_buy_quantity(
        self,
        signal: Signal,
        current_price: float,
    ) -> int:
        """计算买入数量

        优先使用风险管理器的仓位计算，否则使用配置中的固定比例。
        """
        portfolio = self.portfolio_manager.portfolio

        if self.risk_manager is not None:
            quantity = self.risk_manager.calculate_order_quantity(signal, portfolio)
        else:
            # 默认使用固定比例
            target_value = portfolio.total_value * self.config.risk.fixed_ratio
            quantity = int(target_value / current_price)

        # 取整到lot_size
        lot_size = self.config.broker.lot_size
        quantity = (quantity // lot_size) * lot_size

        return quantity

    def _execute_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        current_date: date,
    ) -> None:
        """执行买入"""
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            date=current_date,
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
        )

        # 验证订单
        valid, reason = self.broker.validate_order(
            order, self.portfolio_manager.portfolio
        )
        if not valid:
            logger.debug("买入订单被拒绝: %s %s, 原因: %s", symbol, quantity, reason)
            return

        # 执行订单
        order = self.broker.execute_order(order, price, current_date)
        if order.status == OrderStatus.FILLED:
            self.portfolio_manager.process_buy(order)

    def _execute_sell(
        self,
        symbol: str,
        price: float,
        current_date: date,
    ) -> None:
        """执行卖出（全部可卖数量）"""
        position = self.portfolio_manager.portfolio.get_position(symbol)
        if position is None or position.available_quantity <= 0:
            return

        quantity = position.available_quantity
        # 取整到lot_size
        lot_size = self.config.broker.lot_size
        quantity = (quantity // lot_size) * lot_size
        if quantity <= 0:
            return

        order = Order(
            order_id=str(uuid.uuid4())[:8],
            date=current_date,
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price,
        )

        # 验证订单
        valid, reason = self.broker.validate_order(
            order, self.portfolio_manager.portfolio
        )
        if not valid:
            logger.debug("卖出订单被拒绝: %s %s, 原因: %s", symbol, quantity, reason)
            return

        # 执行订单
        order = self.broker.execute_order(order, price, current_date)
        if order.status == OrderStatus.FILLED:
            trade = self.portfolio_manager.process_sell(order)
            self.trades.append(trade)

    def _check_risk_exits(
        self,
        current_date: date,
        prices: dict[str, float],
    ) -> None:
        """检查风险止损/止盈

        遍历所有持仓，如果触发止损/止盈，生成卖出信号并执行。
        """
        if self.risk_manager is None:
            return

        exit_signals = self.risk_manager.check_exits(
            self.portfolio_manager.portfolio
        )

        for signal in exit_signals:
            symbol = signal.symbol
            price = prices.get(symbol)
            if price is None:
                continue
            logger.info(
                "风险管理触发退出: %s, 原因: %s",
                symbol,
                signal.reason,
            )
            self._execute_sell(symbol, price, current_date)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备行情数据

        确保数据以日期为index并排序。

        Args:
            data: 原始DataFrame

        Returns:
            处理后的DataFrame
        """
        df = data.copy()

        # 如果有date列，设为index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 排序
        df = df.sort_index()

        # 验证必要列
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning("数据缺少列: %s", missing)

        return df

    def _calc_result(self) -> PerformanceResult:
        """计算回测绩效结果"""
        if not self.equity_curve_data:
            return PerformanceMetrics._empty_result()

        # 构建净值曲线Series
        equity_curve = pd.Series(self.equity_curve_data).sort_index()

        return PerformanceMetrics.calculate(
            equity_curve=equity_curve,
            trades=self.trades,
            config=self.config.backtest,
        )
