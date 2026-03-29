"""枚举类型定义 - A股交易相关枚举"""

from enum import Enum


class SignalType(str, Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"       # 市价单
    LIMIT = "limit"         # 限价单


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"       # 待执行
    FILLED = "filled"         # 已成交
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    CANCELLED = "cancelled"   # 已取消
    REJECTED = "rejected"     # 已拒绝


class AdjustType(str, Enum):
    """复权类型"""
    QFQ = "qfq"     # 前复权
    HFQ = "hfq"     # 后复权
    NONE = "none"   # 不复权


class FrequencyType(str, Enum):
    """数据频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    MIN_60 = "60min"


class PositionSizingMethod(str, Enum):
    """仓位管理方法"""
    FIXED_RATIO = "fixed_ratio"         # 固定比例
    FIXED_AMOUNT = "fixed_amount"       # 固定金额
    KELLY = "kelly"                     # Kelly公式
    ATR_BASED = "atr_based"             # ATR自适应


class StopLossMethod(str, Enum):
    """止损方法"""
    FIXED_PERCENT = "fixed_percent"     # 固定百分比
    ATR_TRAILING = "atr_trailing"       # ATR跟踪止损
    SUPPORT_LEVEL = "support_level"     # 支撑位止损


class AssetType(str, Enum):
    """资产类型"""
    STOCK = "stock"             # 个股
    INDEX = "index"             # 指数
    ETF = "etf"                 # ETF基金
    FUND = "fund"               # 场外基金
    BOND = "bond"               # 债券
    FUTURES = "futures"         # 期货
    OPTION = "option"           # 期权
