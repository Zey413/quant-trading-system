"""数据模型定义 - 信号、订单、持仓、组合"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Signal:
    """交易信号"""
    date: date
    symbol: str
    signal_type: str        # "buy" / "sell" / "hold"
    price: float
    strength: float = 1.0   # 信号强度 0~1
    strategy_name: str = ""
    reason: str = ""

    def is_buy(self) -> bool:
        return self.signal_type == "buy"

    def is_sell(self) -> bool:
        return self.signal_type == "sell"


@dataclass
class Order:
    """交易订单"""
    order_id: str
    date: date
    symbol: str
    side: str               # "buy" / "sell"
    order_type: str = "market"
    quantity: int = 0       # 股数 (A股必须是100的整数倍)
    price: float = 0.0      # 限价单价格
    filled_price: float = 0.0
    filled_quantity: int = 0
    commission: float = 0.0
    tax: float = 0.0        # 印花税(仅卖出)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_cost(self) -> float:
        """总成本 = 成交额 + 佣金 + 税"""
        return self.filled_price * self.filled_quantity + self.commission + self.tax


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: int = 0           # 持仓数量
    available_quantity: int = 0  # 可卖数量 (T+1规则)
    avg_cost: float = 0.0       # 平均成本
    current_price: float = 0.0
    buy_date: date | None = None

    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏"""
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """浮动盈亏百分比"""
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class Portfolio:
    """投资组合"""
    cash: float = 1_000_000.0   # 现金 (默认100万)
    positions: dict[str, Position] = field(default_factory=dict)
    initial_capital: float = 1_000_000.0

    @property
    def market_value(self) -> float:
        """持仓总市值"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """总资产 = 现金 + 持仓市值"""
        return self.cash + self.market_value

    @property
    def total_return(self) -> float:
        """总收益率"""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_value - self.initial_capital) / self.initial_capital

    def get_position(self, symbol: str) -> Position | None:
        """获取某只股票的持仓"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """是否持有某只股票"""
        return symbol in self.positions and self.positions[symbol].quantity > 0


@dataclass
class TradeRecord:
    """成交记录"""
    trade_id: str
    date: date
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    tax: float
    pnl: float = 0.0           # 平仓盈亏
    strategy_name: str = ""
