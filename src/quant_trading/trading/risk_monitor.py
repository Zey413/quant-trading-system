"""实时风控监控 - 交易前后的风险检查

负责：
- 定义可扩展的风控规则体系 (RiskRule抽象基类)
- 内置规则: 最大持仓、最大回撤、日亏损限额、集中度
- 交易前风控拦截 (pre-trade check)
- 持仓限额监控
- 日亏损限额监控
- 风险告警通知
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from quant_trading.core.enums import OrderSide
from quant_trading.trading.order_manager import ManagedOrder
from quant_trading.trading.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


# ============================================================
# 告警模型
# ============================================================


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskAlert(BaseModel):
    """风险告警

    Attributes:
        level: 告警级别
        rule_name: 触发规则名称
        message: 告警信息
        timestamp: 告警时间
        data: 附加数据
    """
    level: AlertLevel = AlertLevel.WARNING
    rule_name: str = ""
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    data: dict[str, Any] = Field(default_factory=dict)


class RiskCheckResult(BaseModel):
    """风控检查结果

    Attributes:
        passed: 是否通过
        rule_name: 规则名称
        message: 说明信息
    """
    passed: bool = True
    rule_name: str = ""
    message: str = ""


# 告警回调类型
AlertCallback = Callable[[RiskAlert], None]


# ============================================================
# RiskRule 抽象基类
# ============================================================


class RiskRule(ABC):
    """风控规则抽象基类

    所有风控规则必须实现此接口。

    Attributes:
        name: 规则名称
        enabled: 是否启用
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def check(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> RiskCheckResult:
        """检查订单是否满足此风控规则

        Args:
            order: 待检查的订单
            position_tracker: 持仓跟踪器

        Returns:
            RiskCheckResult: 检查结果
        """
        ...


# ============================================================
# 具体风控规则
# ============================================================


class MaxPositionRule(RiskRule):
    """单只股票最大持仓比例规则

    限制单只股票的持仓市值不超过组合总值的指定比例。

    Attributes:
        max_pct: 最大持仓比例 (0~1)
    """

    def __init__(
        self,
        max_pct: float = 0.3,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="max_position_rule", enabled=enabled)
        self.max_pct = max_pct

    def check(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> RiskCheckResult:
        """检查单只持仓比例"""
        if not self.enabled:
            return RiskCheckResult(passed=True, rule_name=self.name)

        # 仅检查买入
        if order.side != OrderSide.BUY:
            return RiskCheckResult(passed=True, rule_name=self.name)

        total_value = position_tracker.get_portfolio_value()
        if total_value <= 0:
            return RiskCheckResult(
                passed=False,
                rule_name=self.name,
                message="组合总值为零",
            )

        # 当前持仓市值
        position = position_tracker.get_position(order.symbol)
        current_value = position.market_value if position else 0.0

        # 预估新增市值
        additional_value = order.price * order.quantity
        new_position_value = current_value + additional_value
        new_pct = new_position_value / total_value

        if new_pct > self.max_pct:
            return RiskCheckResult(
                passed=False,
                rule_name=self.name,
                message=(
                    f"{order.symbol} 持仓比例将达 {new_pct:.2%}，"
                    f"超过限制 {self.max_pct:.2%}"
                ),
            )

        return RiskCheckResult(passed=True, rule_name=self.name)


class MaxDrawdownRule(RiskRule):
    """最大回撤规则

    当组合回撤超过阈值时，拒绝新的买入订单。

    Attributes:
        max_drawdown: 最大允许回撤 (0~1)
    """

    def __init__(
        self,
        max_drawdown: float = 0.2,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="max_drawdown_rule", enabled=enabled)
        self.max_drawdown = max_drawdown

    def check(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> RiskCheckResult:
        """检查回撤是否超限"""
        if not self.enabled:
            return RiskCheckResult(passed=True, rule_name=self.name)

        # 仅限制买入
        if order.side != OrderSide.BUY:
            return RiskCheckResult(passed=True, rule_name=self.name)

        current_drawdown = position_tracker.get_drawdown()

        if current_drawdown >= self.max_drawdown:
            return RiskCheckResult(
                passed=False,
                rule_name=self.name,
                message=(
                    f"当前回撤 {current_drawdown:.2%} "
                    f"已达上限 {self.max_drawdown:.2%}，禁止买入"
                ),
            )

        return RiskCheckResult(passed=True, rule_name=self.name)


class DailyLossLimitRule(RiskRule):
    """日亏损限额规则

    当日亏损超过限额时，拒绝新的买入订单。

    Attributes:
        max_daily_loss: 最大日亏损金额
        _daily_loss: 当日已实现亏损
        _current_date: 当前日期
    """

    def __init__(
        self,
        max_daily_loss: float = 50000.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="daily_loss_limit_rule", enabled=enabled)
        self.max_daily_loss = max_daily_loss
        self._daily_loss: float = 0.0
        self._current_date: date | None = None

    def record_loss(self, loss: float, trade_date: date | None = None) -> None:
        """记录亏损

        Args:
            loss: 亏损金额（负数表示亏损）
            trade_date: 交易日期
        """
        today = trade_date or date.today()
        if self._current_date != today:
            self._daily_loss = 0.0
            self._current_date = today
        if loss < 0:
            self._daily_loss += abs(loss)

    def reset_daily(self, trade_date: date | None = None) -> None:
        """重置日亏损计数

        Args:
            trade_date: 新交易日期
        """
        self._daily_loss = 0.0
        self._current_date = trade_date or date.today()

    @property
    def daily_loss(self) -> float:
        """当日累计亏损"""
        return self._daily_loss

    def check(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> RiskCheckResult:
        """检查日亏损限额"""
        if not self.enabled:
            return RiskCheckResult(passed=True, rule_name=self.name)

        # 仅限制买入
        if order.side != OrderSide.BUY:
            return RiskCheckResult(passed=True, rule_name=self.name)

        if self._daily_loss >= self.max_daily_loss:
            return RiskCheckResult(
                passed=False,
                rule_name=self.name,
                message=(
                    f"当日亏损 {self._daily_loss:.2f} "
                    f"已达上限 {self.max_daily_loss:.2f}，禁止买入"
                ),
            )

        return RiskCheckResult(passed=True, rule_name=self.name)


class ConcentrationRule(RiskRule):
    """集中度规则

    限制总持仓比例（仓位不超过组合总值的指定比例）。

    Attributes:
        max_concentration: 最大总持仓比例 (0~1)
    """

    def __init__(
        self,
        max_concentration: float = 0.8,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="concentration_rule", enabled=enabled)
        self.max_concentration = max_concentration

    def check(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> RiskCheckResult:
        """检查集中度"""
        if not self.enabled:
            return RiskCheckResult(passed=True, rule_name=self.name)

        # 仅检查买入
        if order.side != OrderSide.BUY:
            return RiskCheckResult(passed=True, rule_name=self.name)

        total_value = position_tracker.get_portfolio_value()
        if total_value <= 0:
            return RiskCheckResult(
                passed=False,
                rule_name=self.name,
                message="组合总值为零",
            )

        market_value = position_tracker.get_market_value()
        additional_value = order.price * order.quantity
        new_concentration = (market_value + additional_value) / total_value

        if new_concentration > self.max_concentration:
            return RiskCheckResult(
                passed=False,
                rule_name=self.name,
                message=(
                    f"总持仓比例将达 {new_concentration:.2%}，"
                    f"超过限制 {self.max_concentration:.2%}"
                ),
            )

        return RiskCheckResult(passed=True, rule_name=self.name)


# ============================================================
# RiskMonitor
# ============================================================


class RiskMonitor:
    """实时风控监控器

    管理多条风控规则，在交易前进行风控检查，
    支持告警回调通知。

    Attributes:
        _rules: 风控规则列表
        _alert_callbacks: 告警回调列表
        _alerts_history: 历史告警记录
        _lock: 线程锁
    """

    def __init__(self) -> None:
        self._rules: list[RiskRule] = []
        self._alert_callbacks: list[AlertCallback] = []
        self._alerts_history: list[RiskAlert] = []
        self._lock = threading.RLock()

    # ----------------------------------------------------------
    # 规则管理
    # ----------------------------------------------------------

    def add_rule(self, rule: RiskRule) -> None:
        """添加风控规则

        Args:
            rule: 风控规则实例
        """
        with self._lock:
            self._rules.append(rule)
        logger.info("添加风控规则: %s", rule.name)

    def remove_rule(self, rule_name: str) -> bool:
        """移除风控规则

        Args:
            rule_name: 规则名称

        Returns:
            是否成功移除
        """
        with self._lock:
            original_len = len(self._rules)
            self._rules = [r for r in self._rules if r.name != rule_name]
            removed = len(self._rules) < original_len
        if removed:
            logger.info("移除风控规则: %s", rule_name)
        return removed

    def get_rule(self, rule_name: str) -> RiskRule | None:
        """获取风控规则

        Args:
            rule_name: 规则名称

        Returns:
            风控规则或None
        """
        with self._lock:
            for rule in self._rules:
                if rule.name == rule_name:
                    return rule
            return None

    def get_rules(self) -> list[RiskRule]:
        """获取所有风控规则"""
        with self._lock:
            return list(self._rules)

    # ----------------------------------------------------------
    # 告警管理
    # ----------------------------------------------------------

    def register_alert_callback(self, callback: AlertCallback) -> None:
        """注册告警回调

        Args:
            callback: 告警回调函数
        """
        with self._lock:
            self._alert_callbacks.append(callback)

    def unregister_alert_callback(self, callback: AlertCallback) -> None:
        """注销告警回调

        Args:
            callback: 告警回调函数
        """
        with self._lock:
            self._alert_callbacks = [
                cb for cb in self._alert_callbacks if cb is not callback
            ]

    def alert(
        self,
        level: AlertLevel,
        rule_name: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> RiskAlert:
        """发出风险告警

        Args:
            level: 告警级别
            rule_name: 规则名称
            message: 告警信息
            data: 附加数据

        Returns:
            RiskAlert: 告警对象
        """
        risk_alert = RiskAlert(
            level=level,
            rule_name=rule_name,
            message=message,
            data=data or {},
        )

        with self._lock:
            self._alerts_history.append(risk_alert)

        # 日志
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.warning)
        log_method("风险告警 [%s] %s: %s", level.value, rule_name, message)

        # 回调通知
        for callback in self._alert_callbacks:
            try:
                callback(risk_alert)
            except Exception:
                logger.exception("告警回调异常: %s", rule_name)

        return risk_alert

    def get_alerts(
        self,
        level: AlertLevel | None = None,
        limit: int = 100,
    ) -> list[RiskAlert]:
        """获取历史告警

        Args:
            level: 告警级别过滤
            limit: 数量限制

        Returns:
            告警列表（最新在前）
        """
        with self._lock:
            alerts = list(self._alerts_history)
        if level:
            alerts = [a for a in alerts if a.level == level]
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]

    # ----------------------------------------------------------
    # 风控检查
    # ----------------------------------------------------------

    def check_pre_trade(
        self,
        order: ManagedOrder,
        position_tracker: PositionTracker,
    ) -> tuple[bool, list[RiskCheckResult]]:
        """交易前风控检查

        逐一检查所有启用的风控规则。任一规则未通过则拒绝交易。

        Args:
            order: 待检查的订单
            position_tracker: 持仓跟踪器

        Returns:
            (all_passed, results): 是否全部通过 + 各规则检查结果
        """
        results: list[RiskCheckResult] = []
        all_passed = True

        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            if not rule.enabled:
                continue
            try:
                result = rule.check(order, position_tracker)
                results.append(result)
                if not result.passed:
                    all_passed = False
                    self.alert(
                        level=AlertLevel.WARNING,
                        rule_name=rule.name,
                        message=result.message,
                        data={
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side.value,
                        },
                    )
            except Exception:
                logger.exception("风控规则检查异常: %s", rule.name)
                results.append(RiskCheckResult(
                    passed=False,
                    rule_name=rule.name,
                    message=f"规则检查异常: {rule.name}",
                ))
                all_passed = False

        return all_passed, results

    def check_position_limit(
        self,
        position_tracker: PositionTracker,
        max_total_pct: float = 0.8,
    ) -> RiskCheckResult:
        """检查总持仓限额

        Args:
            position_tracker: 持仓跟踪器
            max_total_pct: 最大总持仓比例

        Returns:
            RiskCheckResult
        """
        total_value = position_tracker.get_portfolio_value()
        if total_value <= 0:
            return RiskCheckResult(
                passed=False,
                rule_name="position_limit",
                message="组合总值为零",
            )

        market_value = position_tracker.get_market_value()
        pct = market_value / total_value

        if pct > max_total_pct:
            self.alert(
                level=AlertLevel.WARNING,
                rule_name="position_limit",
                message=(
                    f"总持仓比例 {pct:.2%} 超过限制 {max_total_pct:.2%}"
                ),
            )
            return RiskCheckResult(
                passed=False,
                rule_name="position_limit",
                message=f"总持仓比例 {pct:.2%} 超过限制 {max_total_pct:.2%}",
            )

        return RiskCheckResult(passed=True, rule_name="position_limit")

    def check_daily_loss_limit(
        self,
        daily_pnl: float,
        max_daily_loss: float = 50000.0,
    ) -> RiskCheckResult:
        """检查日亏损限额

        Args:
            daily_pnl: 当日盈亏（负数为亏损）
            max_daily_loss: 最大日亏损限额

        Returns:
            RiskCheckResult
        """
        if daily_pnl < 0 and abs(daily_pnl) >= max_daily_loss:
            self.alert(
                level=AlertLevel.CRITICAL,
                rule_name="daily_loss_limit",
                message=(
                    f"当日亏损 {abs(daily_pnl):.2f} "
                    f"已达限额 {max_daily_loss:.2f}"
                ),
                data={"daily_pnl": daily_pnl},
            )
            return RiskCheckResult(
                passed=False,
                rule_name="daily_loss_limit",
                message=(
                    f"当日亏损 {abs(daily_pnl):.2f} 已达限额 {max_daily_loss:.2f}"
                ),
            )

        return RiskCheckResult(passed=True, rule_name="daily_loss_limit")

    # ----------------------------------------------------------
    # 快捷初始化
    # ----------------------------------------------------------

    @classmethod
    def create_default(
        cls,
        max_position_pct: float = 0.3,
        max_drawdown: float = 0.2,
        max_daily_loss: float = 50000.0,
        max_concentration: float = 0.8,
    ) -> RiskMonitor:
        """创建带默认规则的风控监控器

        Args:
            max_position_pct: 单只最大持仓比例
            max_drawdown: 最大回撤
            max_daily_loss: 最大日亏损
            max_concentration: 最大总持仓比例

        Returns:
            RiskMonitor
        """
        monitor = cls()
        monitor.add_rule(MaxPositionRule(max_pct=max_position_pct))
        monitor.add_rule(MaxDrawdownRule(max_drawdown=max_drawdown))
        monitor.add_rule(DailyLossLimitRule(max_daily_loss=max_daily_loss))
        monitor.add_rule(ConcentrationRule(max_concentration=max_concentration))
        return monitor
