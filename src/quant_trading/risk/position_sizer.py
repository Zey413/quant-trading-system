"""仓位管理模块

根据不同的仓位管理方法计算买入数量：
- fixed_ratio: 固定比例法
- kelly: Kelly公式法
- atr_based: ATR自适应法

所有方法均遵守A股规则：
- 数量向下取整到100的整数倍
- 不超过单只最大持仓比例
"""

from __future__ import annotations

import logging

import numpy as np

from quant_trading.core.config import RiskConfig
from quant_trading.core.enums import PositionSizingMethod
from quant_trading.core.models import Portfolio, Signal

logger = logging.getLogger(__name__)

# A股最小交易单位
LOT_SIZE = 100


class PositionSizer:
    """仓位管理器

    根据指定的仓位管理方法和风控限制计算买入数量。

    Attributes:
        config: 风险配置
        _win_rate: 历史胜率（用于Kelly公式）
        _profit_loss_ratio: 历史盈亏比（用于Kelly公式）
        _last_atr: 最近的ATR值（用于ATR法）
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        # Kelly公式所需的参数，需要外部更新
        self._win_rate: float = 0.5
        self._profit_loss_ratio: float = 1.5
        # ATR法所需参数
        self._last_atr: float = 0.0
        self._risk_pct: float = 0.02  # ATR法中每笔交易的风险比例

    def calculate_quantity(
        self,
        signal: Signal,
        portfolio: Portfolio,
        method: str | None = None,
    ) -> int:
        """计算买入数量

        Args:
            signal: 交易信号
            portfolio: 当前组合
            method: 仓位管理方法，None则使用配置中的默认方法

        Returns:
            买入股数（已取整到lot_size，已限制最大持仓比例）
        """
        sizing_method = method or self.config.position_sizing_method
        price = signal.price

        if price <= 0:
            logger.warning("信号价格无效: %.4f", price)
            return 0

        total_value = portfolio.total_value
        if total_value <= 0:
            return 0

        # 根据方法计算原始数量
        if sizing_method == PositionSizingMethod.KELLY:
            quantity = self._kelly_sizing(price, total_value)
        elif sizing_method == PositionSizingMethod.ATR_BASED:
            quantity = self._atr_sizing(price, total_value)
        else:
            # 默认：fixed_ratio
            quantity = self._fixed_ratio_sizing(price, total_value)

        # 应用信号强度调整
        quantity = int(quantity * signal.strength)

        # 应用最大持仓限制
        quantity = self._apply_max_position_limit(
            quantity, signal.symbol, price, portfolio
        )

        # 向下取整到lot_size
        quantity = (quantity // LOT_SIZE) * LOT_SIZE

        # 确保不超过可用现金
        max_by_cash = int(portfolio.cash / price)
        max_by_cash = (max_by_cash // LOT_SIZE) * LOT_SIZE
        quantity = min(quantity, max_by_cash)

        return max(quantity, 0)

    def _fixed_ratio_sizing(self, price: float, total_value: float) -> int:
        """固定比例法

        买入数量 = (总资产 * 固定比例) / 价格
        """
        target_value = total_value * self.config.fixed_ratio
        return int(target_value / price)

    def _kelly_sizing(self, price: float, total_value: float) -> int:
        """Kelly公式法

        Kelly% = W - (1-W) / R
        其中 W = 胜率，R = 盈亏比

        Kelly值可能为负（说明不应该交易），此时返回0。
        为安全起见，实际使用半Kelly（kelly_pct / 2）。
        """
        w = self._win_rate
        r = self._profit_loss_ratio

        if r <= 0:
            return 0

        kelly_pct = w - (1 - w) / r

        if kelly_pct <= 0:
            logger.debug(
                "Kelly公式结果为负: %.4f (胜率=%.2f, 盈亏比=%.2f), 不建议交易",
                kelly_pct,
                w,
                r,
            )
            return 0

        # 使用半Kelly降低风险
        half_kelly = kelly_pct / 2

        # 不超过固定比例上限
        half_kelly = min(half_kelly, self.config.fixed_ratio * 2)

        target_value = total_value * half_kelly
        return int(target_value / price)

    def _atr_sizing(self, price: float, total_value: float) -> int:
        """ATR自适应法

        每股风险 = ATR * 2
        买入数量 = (总资产 * 风险比例) / 每股风险

        如果ATR未设置，回退到固定比例法。
        """
        if self._last_atr <= 0:
            logger.debug("ATR未设置，回退到固定比例法")
            return self._fixed_ratio_sizing(price, total_value)

        risk_per_share = self._last_atr * 2
        if risk_per_share <= 0:
            return 0

        risk_capital = total_value * self._risk_pct
        quantity = int(risk_capital / risk_per_share)
        return quantity

    def _apply_max_position_limit(
        self,
        quantity: int,
        symbol: str,
        price: float,
        portfolio: Portfolio,
    ) -> int:
        """应用最大持仓比例限制

        确保单只股票持仓市值不超过总资产的max_position_pct。
        """
        total_value = portfolio.total_value
        if total_value <= 0:
            return 0

        max_position_value = total_value * self.config.max_position_pct

        # 已有持仓的市值
        existing_value = 0.0
        position = portfolio.get_position(symbol)
        if position is not None:
            existing_value = position.market_value

        # 本次可以增加的市值
        available_value = max_position_value - existing_value
        if available_value <= 0:
            logger.debug(
                "%s 已达最大持仓限制: 当前 %.2f, 上限 %.2f",
                symbol,
                existing_value,
                max_position_value,
            )
            return 0

        max_quantity = int(available_value / price)
        return min(quantity, max_quantity)

    def update_kelly_params(
        self,
        win_rate: float,
        profit_loss_ratio: float,
    ) -> None:
        """更新Kelly公式参数

        Args:
            win_rate: 历史胜率 (0~1)
            profit_loss_ratio: 盈亏比
        """
        self._win_rate = max(0.0, min(1.0, win_rate))
        self._profit_loss_ratio = max(0.0, profit_loss_ratio)

    def update_atr(self, atr: float) -> None:
        """更新ATR值

        Args:
            atr: 最新的ATR值
        """
        self._last_atr = max(0.0, atr)

    def set_risk_pct(self, risk_pct: float) -> None:
        """设置ATR法中的每笔交易风险比例

        Args:
            risk_pct: 风险比例 (0~1)
        """
        self._risk_pct = max(0.0, min(1.0, risk_pct))
