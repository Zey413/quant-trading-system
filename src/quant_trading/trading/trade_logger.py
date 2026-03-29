"""交易日志 - 交易记录与统计

负责：
- 订单、成交、风控事件的日志记录
- 交易数据导出 (CSV/DataFrame)
- 每日交易汇总统计
- 线程安全的并发写入
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from quant_trading.trading.order_manager import ManagedOrder, OrderEvent
from quant_trading.trading.risk_monitor import RiskAlert

logger = logging.getLogger(__name__)


# ============================================================
# 日志模型
# ============================================================


class TradeLogEntry(BaseModel):
    """交易日志条目

    Attributes:
        timestamp: 时间戳
        log_type: 日志类型 (order/trade/risk)
        order_id: 订单ID
        symbol: 股票代码
        side: 方向
        quantity: 数量
        price: 价格
        commission: 佣金
        tax: 印花税
        pnl: 盈亏
        status: 状态
        strategy_name: 策略名称
        message: 备注信息
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    log_type: str = "trade"
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    pnl: float = 0.0
    status: str = ""
    strategy_name: str = ""
    message: str = ""


class RiskLogEntry(BaseModel):
    """风控事件日志条目

    Attributes:
        timestamp: 时间戳
        level: 告警级别
        rule_name: 规则名称
        message: 告警信息
        data: 附加数据
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    level: str = "warning"
    rule_name: str = ""
    message: str = ""
    data: dict = Field(default_factory=dict)


class DailySummary(BaseModel):
    """每日交易汇总

    Attributes:
        trade_date: 交易日期
        total_trades: 总成交笔数
        buy_trades: 买入笔数
        sell_trades: 卖出笔数
        total_volume: 总成交量 (股数)
        total_turnover: 总成交额
        total_commission: 总佣金
        total_tax: 总税费
        realized_pnl: 已实现盈亏
        risk_events: 风控事件数
    """
    trade_date: date
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    total_volume: int = 0
    total_turnover: float = 0.0
    total_commission: float = 0.0
    total_tax: float = 0.0
    realized_pnl: float = 0.0
    risk_events: int = 0


# ============================================================
# TradeLogger
# ============================================================


class TradeLogger:
    """交易日志器

    线程安全的交易日志管理器，支持订单、成交、风控事件的记录和查询。

    Attributes:
        _trade_logs: 交易日志列表
        _risk_logs: 风控日志列表
        _lock: 线程锁
    """

    def __init__(self) -> None:
        self._trade_logs: list[TradeLogEntry] = []
        self._risk_logs: list[RiskLogEntry] = []
        self._lock = threading.RLock()

    # ----------------------------------------------------------
    # 日志记录
    # ----------------------------------------------------------

    def log_order(self, order: ManagedOrder, message: str = "") -> None:
        """记录订单事件

        Args:
            order: 订单对象
            message: 备注信息
        """
        entry = TradeLogEntry(
            log_type="order",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price,
            commission=order.commission,
            tax=order.tax,
            status=order.status.value,
            strategy_name=order.strategy_name,
            message=message or f"订单 {order.status.value}",
        )

        with self._lock:
            self._trade_logs.append(entry)

        logger.debug(
            "日志-订单: %s %s %s %d股 @ %.4f [%s]",
            order.order_id,
            order.side.value,
            order.symbol,
            order.quantity,
            order.price,
            order.status.value,
        )

    def log_trade(
        self,
        order: ManagedOrder,
        filled_price: float,
        filled_quantity: int,
        commission: float = 0.0,
        tax: float = 0.0,
        pnl: float = 0.0,
        message: str = "",
    ) -> None:
        """记录成交事件

        Args:
            order: 订单对象
            filled_price: 成交价格
            filled_quantity: 成交数量
            commission: 佣金
            tax: 印花税
            pnl: 盈亏
            message: 备注信息
        """
        entry = TradeLogEntry(
            log_type="trade",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=filled_quantity,
            price=filled_price,
            commission=commission,
            tax=tax,
            pnl=pnl,
            status="filled",
            strategy_name=order.strategy_name,
            message=message or "成交",
        )

        with self._lock:
            self._trade_logs.append(entry)

        logger.info(
            "日志-成交: %s %s %s %d股 @ %.4f, PnL=%.2f",
            order.order_id,
            order.side.value,
            order.symbol,
            filled_quantity,
            filled_price,
            pnl,
        )

    def log_risk_event(
        self,
        alert: RiskAlert | None = None,
        level: str = "warning",
        rule_name: str = "",
        message: str = "",
        data: dict | None = None,
    ) -> None:
        """记录风控事件

        Args:
            alert: 风险告警对象（如果有）
            level: 告警级别
            rule_name: 规则名称
            message: 告警信息
            data: 附加数据
        """
        if alert is not None:
            entry = RiskLogEntry(
                level=alert.level.value,
                rule_name=alert.rule_name,
                message=alert.message,
                data=alert.data,
            )
        else:
            entry = RiskLogEntry(
                level=level,
                rule_name=rule_name,
                message=message,
                data=data or {},
            )

        with self._lock:
            self._risk_logs.append(entry)

        logger.info(
            "日志-风控: [%s] %s: %s",
            entry.level,
            entry.rule_name,
            entry.message,
        )

    def on_order_event(self, event: OrderEvent) -> None:
        """OrderManager 事件回调

        可直接注册到 OrderManager 的回调系统中。

        Args:
            event: 订单事件
        """
        self.log_order(
            event.order,
            message=f"事件: {event.event_type.value}",
        )

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_trade_logs(
        self,
        symbol: str | None = None,
        log_type: str | None = None,
        limit: int = 200,
    ) -> list[TradeLogEntry]:
        """获取交易日志

        Args:
            symbol: 股票代码过滤
            log_type: 日志类型过滤 (order/trade)
            limit: 数量限制

        Returns:
            交易日志列表（最新在前）
        """
        with self._lock:
            logs = list(self._trade_logs)

        if symbol:
            logs = [l for l in logs if l.symbol == symbol]
        if log_type:
            logs = [l for l in logs if l.log_type == log_type]

        logs.sort(key=lambda l: l.timestamp, reverse=True)
        return logs[:limit]

    def get_risk_logs(self, limit: int = 100) -> list[RiskLogEntry]:
        """获取风控日志

        Args:
            limit: 数量限制

        Returns:
            风控日志列表（最新在前）
        """
        with self._lock:
            logs = list(self._risk_logs)
        logs.sort(key=lambda l: l.timestamp, reverse=True)
        return logs[:limit]

    # ----------------------------------------------------------
    # 每日汇总
    # ----------------------------------------------------------

    def get_daily_summary(self, trade_date: date | None = None) -> DailySummary:
        """获取每日交易汇总

        Args:
            trade_date: 交易日期（None为今天）

        Returns:
            DailySummary: 日汇总
        """
        target_date = trade_date or date.today()

        with self._lock:
            trade_logs = [
                l for l in self._trade_logs
                if l.log_type == "trade"
                and l.timestamp.date() == target_date
            ]
            risk_count = sum(
                1 for l in self._risk_logs
                if l.timestamp.date() == target_date
            )

        total_trades = len(trade_logs)
        buy_trades = sum(1 for l in trade_logs if l.side == "buy")
        sell_trades = sum(1 for l in trade_logs if l.side == "sell")
        total_volume = sum(l.quantity for l in trade_logs)
        total_turnover = sum(l.price * l.quantity for l in trade_logs)
        total_commission = sum(l.commission for l in trade_logs)
        total_tax = sum(l.tax for l in trade_logs)
        realized_pnl = sum(l.pnl for l in trade_logs)

        return DailySummary(
            trade_date=target_date,
            total_trades=total_trades,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            total_volume=total_volume,
            total_turnover=round(total_turnover, 2),
            total_commission=round(total_commission, 2),
            total_tax=round(total_tax, 2),
            realized_pnl=round(realized_pnl, 2),
            risk_events=risk_count,
        )

    # ----------------------------------------------------------
    # 导出
    # ----------------------------------------------------------

    def export_trades(self, filepath: str | Path) -> int:
        """导出交易记录到CSV文件

        Args:
            filepath: 输出文件路径

        Returns:
            导出的记录数
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            trade_logs = [
                l for l in self._trade_logs if l.log_type == "trade"
            ]

        if not trade_logs:
            logger.warning("没有交易记录可导出")
            return 0

        fieldnames = [
            "timestamp", "order_id", "symbol", "side", "quantity",
            "price", "commission", "tax", "pnl", "strategy_name", "message",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in trade_logs:
                writer.writerow({
                    "timestamp": entry.timestamp.isoformat(),
                    "order_id": entry.order_id,
                    "symbol": entry.symbol,
                    "side": entry.side,
                    "quantity": entry.quantity,
                    "price": entry.price,
                    "commission": entry.commission,
                    "tax": entry.tax,
                    "pnl": entry.pnl,
                    "strategy_name": entry.strategy_name,
                    "message": entry.message,
                })

        logger.info("导出 %d 条交易记录到 %s", len(trade_logs), filepath)
        return len(trade_logs)

    def export_risk_events(self, filepath: str | Path) -> int:
        """导出风控事件到CSV文件

        Args:
            filepath: 输出文件路径

        Returns:
            导出的记录数
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            risk_logs = list(self._risk_logs)

        if not risk_logs:
            logger.warning("没有风控事件可导出")
            return 0

        fieldnames = ["timestamp", "level", "rule_name", "message", "data"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in risk_logs:
                writer.writerow({
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level,
                    "rule_name": entry.rule_name,
                    "message": entry.message,
                    "data": str(entry.data),
                })

        logger.info("导出 %d 条风控事件到 %s", len(risk_logs), filepath)
        return len(risk_logs)

    # ----------------------------------------------------------
    # 统计
    # ----------------------------------------------------------

    @property
    def trade_count(self) -> int:
        """成交记录数"""
        with self._lock:
            return sum(1 for l in self._trade_logs if l.log_type == "trade")

    @property
    def order_count(self) -> int:
        """订单记录数"""
        with self._lock:
            return sum(1 for l in self._trade_logs if l.log_type == "order")

    @property
    def risk_event_count(self) -> int:
        """风控事件数"""
        with self._lock:
            return len(self._risk_logs)

    def clear(self) -> None:
        """清空所有日志"""
        with self._lock:
            self._trade_logs.clear()
            self._risk_logs.clear()
