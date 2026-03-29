"""质量因子

提供:
- ROEFactor: 净资产收益率因子
- ROAFactor: 总资产收益率因子

质量因子反映公司盈利能力，ROE/ROA 越高表示公司质量越好。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.factor.base import Factor, FactorDirection, FactorRegistry


@FactorRegistry.register("roe")
class ROEFactor(Factor):
    """净资产收益率因子 (ROE)

    ROE = 净利润 / 净资产

    ROE 越高表示公司盈利质量越好，预期收益越高。

    Parameters
    ----------
    ttm : bool
        是否使用 TTM（滚动12个月）数据，默认 True
    name : str
        因子名称
    """

    def __init__(self, ttm: bool = True, name: str = "") -> None:
        super().__init__(
            name=name or ("roe_ttm" if ttm else "roe"),
            direction=FactorDirection.POSITIVE,
        )
        self.ttm = ttm

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 ROE 因子值

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 ROE 相关列，或 ``net_profit`` 和 ``equity`` 列

        Returns
        -------
        pd.Series
            ROE 因子值
        """
        roe_col = self._find_roe_column(data)
        if roe_col is not None:
            return data[roe_col].copy()

        # 如果没有直接的 ROE 列，尝试从财务数据计算
        if "net_profit" in data.columns and "equity" in data.columns:
            equity = data["equity"]
            roe = pd.Series(index=data.index, dtype=float)
            valid_mask = equity > 0
            roe[valid_mask] = data.loc[valid_mask, "net_profit"] / equity[valid_mask]
            return roe

        raise KeyError(
            f"DataFrame 中未找到 ROE 相关列，可用列: {list(data.columns)}"
        )

    def _find_roe_column(self, data: pd.DataFrame) -> str | None:
        """查找 ROE 列名"""
        candidates = (
            ["roe_ttm", "roe", "ROE_TTM", "ROE"]
            if self.ttm
            else ["roe", "roe_ttm", "ROE", "ROE_TTM"]
        )
        for col in candidates:
            if col in data.columns:
                return col
        return None

    def get_params(self) -> dict:
        return {"ttm": self.ttm}


@FactorRegistry.register("roa")
class ROAFactor(Factor):
    """总资产收益率因子 (ROA)

    ROA = 净利润 / 总资产

    ROA 越高表示公司资产利用效率越高，预期收益越高。

    Parameters
    ----------
    ttm : bool
        是否使用 TTM（滚动12个月）数据，默认 True
    name : str
        因子名称
    """

    def __init__(self, ttm: bool = True, name: str = "") -> None:
        super().__init__(
            name=name or ("roa_ttm" if ttm else "roa"),
            direction=FactorDirection.POSITIVE,
        )
        self.ttm = ttm

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 ROA 因子值

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 ROA 相关列，或 ``net_profit`` 和 ``total_assets`` 列

        Returns
        -------
        pd.Series
            ROA 因子值
        """
        roa_col = self._find_roa_column(data)
        if roa_col is not None:
            return data[roa_col].copy()

        # 如果没有直接的 ROA 列，尝试从财务数据计算
        if "net_profit" in data.columns and "total_assets" in data.columns:
            total_assets = data["total_assets"]
            roa = pd.Series(index=data.index, dtype=float)
            valid_mask = total_assets > 0
            roa[valid_mask] = (
                data.loc[valid_mask, "net_profit"] / total_assets[valid_mask]
            )
            return roa

        raise KeyError(
            f"DataFrame 中未找到 ROA 相关列，可用列: {list(data.columns)}"
        )

    def _find_roa_column(self, data: pd.DataFrame) -> str | None:
        """查找 ROA 列名"""
        candidates = (
            ["roa_ttm", "roa", "ROA_TTM", "ROA"]
            if self.ttm
            else ["roa", "roa_ttm", "ROA", "ROA_TTM"]
        )
        for col in candidates:
            if col in data.columns:
                return col
        return None

    def get_params(self) -> dict:
        return {"ttm": self.ttm}
