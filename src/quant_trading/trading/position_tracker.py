"""持仓跟踪器 - 持仓管理与盈亏计算

负责：
- 持仓的实时更新与跟踪
- 浮动盈亏、已实现盈亏计算
- 组合市值计算
- 盈亏报告生成
- 线程安全的并发访问
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from quant_trading.core.enums import OrderSide

logger = logging.getLogger(__name__)


# ============================================================
# 持仓与盈亏模型
# ============================================================


class TrackedPosition(BaseModel):
    """受跟踪的持仓 - Pydantic模型

    Attributes:
        symbol: 股票代码
        quantity: 持仓数量
        available_quantity: 可卖数量 (T+1)
        avg_cost: 平均成本
        current_price: 当前价格
        realized_pnl: 已实现盈亏（累计）
        buy_date: 首次买入日期
        last_update: 最后更新时间
    """
    symbol: str
    quantity: int = 0
    available_quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    buy_date: Optional[date] = None
    last_update: datetime = Field(default_factory=datetime.now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def market_value(self) -> float:
        """持仓市值"""
        return self.quantity * self.current_price

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏"""
        return (self.current_price - self.avg_cost) * self.quantity

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealized_pnl_pct(self) -> float:
        """浮动盈亏百分比"""
        if self.avg_cost == 0 or self.quantity == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_pnl(self) -> float:
        """总盈亏 = 已实现 + 未实现"""
        return self.realized_pnl + self.unrealized_pnl

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_basis(self) -> float:
        """持仓成本 = 均价 * 数量"""
        return self.avg_cost * self.quantity


class PnLReport(BaseModel):
    """盈亏报告

    Attributes:
        timestamp: 报告时间
        total_market_value: 总持仓市值
        total_unrealized_pnl: 总浮动盈亏
        total_realized_pnl: 总已实现盈亏
        total_pnl: 总盈亏
        cash: 可用现金
        portfolio_value: 组合总值 (现金+市值)
        position_count: 持仓数量
        positions: 各持仓明细
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    total_market_value: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    total_pnl: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    position_count: int = 0
    positions: dict[str, TrackedPosition] = Field(default_factory=dict)


class PositionUpdateResult(BaseModel):
    """持仓更新结果"""
    symbol: str
    side: str
    quantity: int
    price: float
    realized_pnl: float = 0.0
    new_position: Optional[TrackedPosition] = None


# ============================================================
# PositionTracker
# ============================================================


class PositionTracker:
    """持仓跟踪器

    线程安全的持仓管理器，跟踪所有持仓的状态和盈亏。

    Attributes:
        _positions: 持仓字典 {symbol: TrackedPosition}
        _cash: 可用现金
        _initial_capital: 初始资金
        _pending_available: T+1 待释放的可卖数量
        _lock: 线程锁
        _peak_value: 历史最高组合价值（用于回撤计算）
    """

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        """初始化

        Args:
            initial_capital: 初始资金
        """
        self._positions: dict[str, TrackedPosition] = {}
        self._cash: float = initial_capital
        self._initial_capital: float = initial_capital
        self._pending_available: dict[str, int] = {}
        self._lock = threading.RLock()
        self._peak_value: float = initial_capital

    # ----------------------------------------------------------
    # 持仓更新
    # ----------------------------------------------------------

    def update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        commission: float = 0.0,
        tax: float = 0.0,
        trade_date: date | None = None,
    ) -> PositionUpdateResult:
        """更新持仓（买入增仓 / 卖出减仓）

        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 成交数量
            price: 成交价格
            commission: 佣金
            tax: 印花税
            trade_date: 交易日期

        Returns:
            PositionUpdateResult: 更新结果（含已实现盈亏）

        Raises:
            ValueError: 参数不合法或持仓不足
        """
        if quantity <= 0:
            raise ValueError(f"成交数量必须大于0: {quantity}")
        if price <= 0:
            raise ValueError(f"成交价格必须大于0: {price}")

        with self._lock:
            if side == OrderSide.BUY:
                return self._process_buy(
                    symbol, quantity, price, commission, tax, trade_date
                )
            else:
                return self._process_sell(
                    symbol, quantity, price, commission, tax
                )

    def _process_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        commission: float,
        tax: float,
        trade_date: date | None,
    ) -> PositionUpdateResult:
        """处理买入"""
        total_cost = price * quantity + commission + tax
        self._cash -= total_cost

        position = self._positions.get(symbol)

        if position is not None and position.quantity > 0:
            # 加仓: 更新平均成本
            old_total = position.avg_cost * position.quantity
            new_total = price * quantity
            total_qty = position.quantity + quantity
            position.avg_cost = (old_total + new_total) / total_qty
            position.quantity = total_qty
            position.current_price = price
            position.last_update = datetime.now()
        else:
            # 新建仓位
            position = TrackedPosition(
                symbol=symbol,
                quantity=quantity,
                available_quantity=0,
                avg_cost=price,
                current_price=price,
                buy_date=trade_date or date.today(),
            )
            self._positions[symbol] = position

        # T+1: 当日买入不可卖
        self._pending_available[symbol] = (
            self._pending_available.get(symbol, 0) + quantity
        )

        logger.debug(
            "买入 %s %d股 @ %.4f, 花费 %.2f, 剩余现金 %.2f",
            symbol, quantity, price, total_cost, self._cash,
        )

        return PositionUpdateResult(
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            new_position=position,
        )

    def _process_sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        commission: float,
        tax: float,
    ) -> PositionUpdateResult:
        """处理卖出"""
        position = self._positions.get(symbol)
        if position is None or position.quantity <= 0:
            raise ValueError(f"未持有 {symbol}，无法卖出")
        if quantity > position.available_quantity:
            raise ValueError(
                f"{symbol} 可卖数量不足: 需要 {quantity}，"
                f"可卖 {position.available_quantity}"
            )

        # 卖出收入
        turnover = price * quantity
        net_income = turnover - commission - tax
        self._cash += net_income

        # 已实现盈亏
        realized_pnl = (price - position.avg_cost) * quantity - commission - tax
        position.realized_pnl += realized_pnl

        # 减仓
        position.quantity -= quantity
        position.available_quantity -= quantity
        position.current_price = price
        position.last_update = datetime.now()

        # 清仓则移除
        result_position: TrackedPosition | None = position
        if position.quantity <= 0:
            del self._positions[symbol]
            self._pending_available.pop(symbol, None)
            result_position = None

        logger.debug(
            "卖出 %s %d股 @ %.4f, 收入 %.2f, PnL=%.2f",
            symbol, quantity, price, net_income, realized_pnl,
        )

        return PositionUpdateResult(
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            realized_pnl=realized_pnl,
            new_position=result_position,
        )

    # ----------------------------------------------------------
    # 价格更新与T+1
    # ----------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """更新某只股票的当前价格

        Args:
            symbol: 股票代码
            price: 当前价格
        """
        with self._lock:
            position = self._positions.get(symbol)
            if position is not None:
                position.current_price = price
                position.last_update = datetime.now()

    def update_prices(self, prices: dict[str, float]) -> None:
        """批量更新价格

        Args:
            prices: {symbol: price} 字典
        """
        with self._lock:
            for symbol, price in prices.items():
                position = self._positions.get(symbol)
                if position is not None:
                    position.current_price = price
                    position.last_update = datetime.now()

    def new_trading_day(self) -> None:
        """处理新交易日 - T+1规则

        将前一日买入的数量转入可卖数量。
        """
        with self._lock:
            for symbol, qty in self._pending_available.items():
                position = self._positions.get(symbol)
                if position is not None:
                    position.available_quantity += qty
            self._pending_available.clear()

            # 更新峰值
            current_value = self.get_portfolio_value()
            if current_value > self._peak_value:
                self._peak_value = current_value

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_position(self, symbol: str) -> TrackedPosition | None:
        """获取某只股票的持仓

        Args:
            symbol: 股票代码

        Returns:
            TrackedPosition 或 None
        """
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, TrackedPosition]:
        """获取所有持仓

        Returns:
            {symbol: TrackedPosition} 字典的副本
        """
        with self._lock:
            return dict(self._positions)

    def has_position(self, symbol: str) -> bool:
        """是否持有某只股票"""
        with self._lock:
            pos = self._positions.get(symbol)
            return pos is not None and pos.quantity > 0

    def get_portfolio_value(self) -> float:
        """获取组合总值 = 现金 + 持仓市值"""
        with self._lock:
            market_value = sum(
                p.market_value for p in self._positions.values()
            )
            return self._cash + market_value

    def get_market_value(self) -> float:
        """获取持仓总市值"""
        with self._lock:
            return sum(p.market_value for p in self._positions.values())

    @property
    def cash(self) -> float:
        """可用现金"""
        with self._lock:
            return self._cash

    @property
    def initial_capital(self) -> float:
        """初始资金"""
        return self._initial_capital

    @property
    def peak_value(self) -> float:
        """历史最高组合价值"""
        with self._lock:
            return self._peak_value

    def get_drawdown(self) -> float:
        """当前回撤百分比 (0~1)"""
        with self._lock:
            current = self.get_portfolio_value()
            if self._peak_value <= 0:
                return 0.0
            return max(0.0, (self._peak_value - current) / self._peak_value)

    # ----------------------------------------------------------
    # 盈亏报告
    # ----------------------------------------------------------

    def get_pnl(self) -> PnLReport:
        """生成盈亏报告

        Returns:
            PnLReport: 完整的盈亏报告
        """
        with self._lock:
            total_market_value = 0.0
            total_unrealized = 0.0
            total_realized = 0.0
            positions_copy: dict[str, TrackedPosition] = {}

            for symbol, pos in self._positions.items():
                total_market_value += pos.market_value
                total_unrealized += pos.unrealized_pnl
                total_realized += pos.realized_pnl
                positions_copy[symbol] = pos.model_copy()

            portfolio_value = self._cash + total_market_value
            total_pnl = total_realized + total_unrealized

            return PnLReport(
                total_market_value=round(total_market_value, 2),
                total_unrealized_pnl=round(total_unrealized, 2),
                total_realized_pnl=round(total_realized, 2),
                total_pnl=round(total_pnl, 2),
                cash=round(self._cash, 2),
                portfolio_value=round(portfolio_value, 2),
                position_count=len(self._positions),
                positions=positions_copy,
            )

    def get_return_pct(self) -> float:
        """总收益率"""
        if self._initial_capital <= 0:
            return 0.0
        return (self.get_portfolio_value() - self._initial_capital) / self._initial_capital
