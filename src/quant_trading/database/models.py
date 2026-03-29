"""数据库表结构定义 - 使用 dataclass 定义所有表模型

每个 dataclass 对应一张 SQLite 表，字段与列一一映射。
与 core.models 中的 Pydantic 模型兼容，提供互转方法。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any


# ---------------------------------------------------------------------------
# 行情数据
# ---------------------------------------------------------------------------

@dataclass
class StockDaily:
    """日线行情数据

    对应表: stock_daily
    主键: (symbol, trade_date)
    """
    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0           # 成交额
    turnover: float = 0.0         # 换手率
    pct_change: float = 0.0       # 涨跌幅 (%)
    pre_close: float = 0.0        # 前收盘价
    adj_factor: float = 1.0       # 复权因子
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trade_date"] = self.trade_date.isoformat()
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_row(cls, row: tuple) -> StockDaily:
        return cls(
            symbol=row[0],
            trade_date=date.fromisoformat(row[1]),
            open=row[2],
            high=row[3],
            low=row[4],
            close=row[5],
            volume=row[6],
            amount=row[7],
            turnover=row[8],
            pct_change=row[9],
            pre_close=row[10],
            adj_factor=row[11],
            created_at=datetime.fromisoformat(row[12]),
        )


@dataclass
class StockInfo:
    """股票基本信息

    对应表: stock_info
    主键: symbol
    """
    symbol: str
    name: str
    market: str = ""              # SH / SZ / BJ
    industry: str = ""
    sector: str = ""
    list_date: date | None = None
    delist_date: date | None = None
    is_active: bool = True
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["list_date"] = self.list_date.isoformat() if self.list_date else None
        d["delist_date"] = self.delist_date.isoformat() if self.delist_date else None
        d["updated_at"] = self.updated_at.isoformat()
        d["is_active"] = int(self.is_active)
        return d

    @classmethod
    def from_row(cls, row: tuple) -> StockInfo:
        return cls(
            symbol=row[0],
            name=row[1],
            market=row[2],
            industry=row[3],
            sector=row[4],
            list_date=date.fromisoformat(row[5]) if row[5] else None,
            delist_date=date.fromisoformat(row[6]) if row[6] else None,
            is_active=bool(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
        )


# ---------------------------------------------------------------------------
# 交易记录
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """成交记录

    对应表: trade_record
    主键: trade_id
    """
    trade_id: str
    date: date
    symbol: str
    side: str                     # "buy" / "sell"
    quantity: int
    price: float
    commission: float = 0.0
    tax: float = 0.0
    pnl: float = 0.0             # 平仓盈亏
    strategy_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_cost(self) -> float:
        """总成本 = 成交额 + 佣金 + 税"""
        return self.price * self.quantity + self.commission + self.tax

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_row(cls, row: tuple) -> TradeRecord:
        return cls(
            trade_id=row[0],
            date=date.fromisoformat(row[1]),
            symbol=row[2],
            side=row[3],
            quantity=row[4],
            price=row[5],
            commission=row[6],
            tax=row[7],
            pnl=row[8],
            strategy_name=row[9],
            created_at=datetime.fromisoformat(row[10]),
        )

    @classmethod
    def from_core_model(cls, record) -> TradeRecord:
        """从 core.models.TradeRecord 转换"""
        return cls(
            trade_id=record.trade_id,
            date=record.date,
            symbol=record.symbol,
            side=record.side,
            quantity=record.quantity,
            price=record.price,
            commission=record.commission,
            tax=record.tax,
            pnl=record.pnl,
            strategy_name=record.strategy_name,
        )


# ---------------------------------------------------------------------------
# 回测结果
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """回测结果

    对应表: backtest_result
    主键: result_id (自增)
    """
    result_id: int | None = None
    strategy_name: str = ""
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0     # 总收益率 (%)
    annual_return: float = 0.0    # 年化收益率 (%)
    max_drawdown: float = 0.0     # 最大回撤 (%)
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0         # 胜率 (%)
    total_trades: int = 0
    params_json: str = "{}"       # 策略参数 JSON
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat() if self.start_date else None
        d["end_date"] = self.end_date.isoformat() if self.end_date else None
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_row(cls, row: tuple) -> BacktestResult:
        return cls(
            result_id=row[0],
            strategy_name=row[1],
            start_date=date.fromisoformat(row[2]) if row[2] else None,
            end_date=date.fromisoformat(row[3]) if row[3] else None,
            initial_capital=row[4],
            final_value=row[5],
            total_return=row[6],
            annual_return=row[7],
            max_drawdown=row[8],
            sharpe_ratio=row[9],
            win_rate=row[10],
            total_trades=row[11],
            params_json=row[12],
            created_at=datetime.fromisoformat(row[13]),
        )


# ---------------------------------------------------------------------------
# 策略参数
# ---------------------------------------------------------------------------

@dataclass
class StrategyParam:
    """策略参数快照

    对应表: strategy_param
    主键: param_id (自增)
    """
    param_id: int | None = None
    strategy_name: str = ""
    params_json: str = "{}"
    description: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_active"] = int(self.is_active)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_row(cls, row: tuple) -> StrategyParam:
        return cls(
            param_id=row[0],
            strategy_name=row[1],
            params_json=row[2],
            description=row[3],
            is_active=bool(row[4]),
            created_at=datetime.fromisoformat(row[5]),
        )


# ---------------------------------------------------------------------------
# 投资组合快照
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    """投资组合快照

    对应表: portfolio_snapshot
    主键: snapshot_id (自增)
    """
    snapshot_id: int | None = None
    snapshot_date: date | None = None
    cash: float = 0.0
    market_value: float = 0.0
    total_value: float = 0.0
    total_return: float = 0.0     # 总收益率 (%)
    positions_json: str = "{}"    # 持仓明细 JSON
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["snapshot_date"] = (
            self.snapshot_date.isoformat() if self.snapshot_date else None
        )
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_row(cls, row: tuple) -> PortfolioSnapshot:
        return cls(
            snapshot_id=row[0],
            snapshot_date=date.fromisoformat(row[1]) if row[1] else None,
            cash=row[2],
            market_value=row[3],
            total_value=row[4],
            total_return=row[5],
            positions_json=row[6],
            created_at=datetime.fromisoformat(row[7]),
        )


# ---------------------------------------------------------------------------
# 自选股
# ---------------------------------------------------------------------------

@dataclass
class WatchlistItem:
    """自选股

    对应表: watchlist
    主键: item_id (自增)
    唯一约束: symbol
    """
    item_id: int | None = None
    symbol: str = ""
    name: str = ""
    notes: str = ""
    target_price: float | None = None
    stop_loss_price: float | None = None
    added_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["added_at"] = self.added_at.isoformat()
        return d

    @classmethod
    def from_row(cls, row: tuple) -> WatchlistItem:
        return cls(
            item_id=row[0],
            symbol=row[1],
            name=row[2],
            notes=row[3],
            target_price=row[4],
            stop_loss_price=row[5],
            added_at=datetime.fromisoformat(row[6]),
        )


# ---------------------------------------------------------------------------
# 告警
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """告警/提醒

    对应表: alert
    主键: alert_id (自增)
    """
    alert_id: int | None = None
    symbol: str = ""
    alert_type: str = ""          # "price_above" / "price_below" / "pct_change" / "custom"
    threshold: float = 0.0
    message: str = ""
    is_triggered: bool = False
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    triggered_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_triggered"] = int(self.is_triggered)
        d["is_active"] = int(self.is_active)
        d["created_at"] = self.created_at.isoformat()
        d["triggered_at"] = (
            self.triggered_at.isoformat() if self.triggered_at else None
        )
        return d

    @classmethod
    def from_row(cls, row: tuple) -> Alert:
        return cls(
            alert_id=row[0],
            symbol=row[1],
            alert_type=row[2],
            threshold=row[3],
            message=row[4],
            is_triggered=bool(row[5]),
            is_active=bool(row[6]),
            created_at=datetime.fromisoformat(row[7]),
            triggered_at=(
                datetime.fromisoformat(row[8]) if row[8] else None
            ),
        )
