"""价值因子

提供:
- PEFactor: 市盈率因子 (Price/Earnings, 取倒数 EP 作为因子值)
- PBFactor: 市净率因子 (Price/Book, 取倒数 BP 作为因子值)
- PSFactor: 市销率因子 (Price/Sales, 取倒数 SP 作为因子值)

注: 价值因子通常取估值指标的倒数，使因子值越大表示越便宜（越有价值）。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.factor.base import Factor, FactorDirection, FactorRegistry


@FactorRegistry.register("pe")
class PEFactor(Factor):
    """市盈率价值因子 (EP = 1/PE)

    EP 越高表示股票越便宜（被低估），预期收益越高。

    Parameters
    ----------
    inverse : bool
        是否取倒数（EP），默认 True。
        若为 False，则直接使用 PE 值（方向为 NEGATIVE）。
    name : str
        因子名称
    """

    def __init__(self, inverse: bool = True, name: str = "") -> None:
        direction = FactorDirection.POSITIVE if inverse else FactorDirection.NEGATIVE
        super().__init__(
            name=name or ("ep" if inverse else "pe"),
            direction=direction,
        )
        self.inverse = inverse

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 PE/EP 因子值

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 ``pe`` 或 ``pe_ttm`` 列

        Returns
        -------
        pd.Series
            EP（1/PE）或 PE 值
        """
        pe_col = self._find_pe_column(data)
        pe = data[pe_col].copy()

        if self.inverse:
            # EP = 1 / PE，排除 PE <= 0 的情况（亏损股）
            ep = pd.Series(index=pe.index, dtype=float)
            valid_mask = pe > 0
            ep[valid_mask] = 1.0 / pe[valid_mask]
            return ep
        return pe

    @staticmethod
    def _find_pe_column(data: pd.DataFrame) -> str:
        """查找 PE 列名"""
        for col in ["pe_ttm", "pe", "PE_TTM", "PE"]:
            if col in data.columns:
                return col
        raise KeyError(
            f"DataFrame 中未找到 PE 列，可用列: {list(data.columns)}"
        )

    def get_params(self) -> dict:
        return {"inverse": self.inverse}


@FactorRegistry.register("pb")
class PBFactor(Factor):
    """市净率价值因子 (BP = 1/PB)

    BP 越高表示股票越便宜，预期收益越高。

    Parameters
    ----------
    inverse : bool
        是否取倒数（BP），默认 True。
    name : str
        因子名称
    """

    def __init__(self, inverse: bool = True, name: str = "") -> None:
        direction = FactorDirection.POSITIVE if inverse else FactorDirection.NEGATIVE
        super().__init__(
            name=name or ("bp" if inverse else "pb"),
            direction=direction,
        )
        self.inverse = inverse

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 PB/BP 因子值

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 ``pb`` 或 ``pb_mrq`` 列

        Returns
        -------
        pd.Series
            BP（1/PB）或 PB 值
        """
        pb_col = self._find_pb_column(data)
        pb = data[pb_col].copy()

        if self.inverse:
            bp = pd.Series(index=pb.index, dtype=float)
            valid_mask = pb > 0
            bp[valid_mask] = 1.0 / pb[valid_mask]
            return bp
        return pb

    @staticmethod
    def _find_pb_column(data: pd.DataFrame) -> str:
        """查找 PB 列名"""
        for col in ["pb_mrq", "pb", "PB_MRQ", "PB"]:
            if col in data.columns:
                return col
        raise KeyError(
            f"DataFrame 中未找到 PB 列，可用列: {list(data.columns)}"
        )

    def get_params(self) -> dict:
        return {"inverse": self.inverse}


@FactorRegistry.register("ps")
class PSFactor(Factor):
    """市销率价值因子 (SP = 1/PS)

    SP 越高表示股票越便宜，预期收益越高。

    Parameters
    ----------
    inverse : bool
        是否取倒数（SP），默认 True。
    name : str
        因子名称
    """

    def __init__(self, inverse: bool = True, name: str = "") -> None:
        direction = FactorDirection.POSITIVE if inverse else FactorDirection.NEGATIVE
        super().__init__(
            name=name or ("sp" if inverse else "ps"),
            direction=direction,
        )
        self.inverse = inverse

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 PS/SP 因子值

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 ``ps`` 或 ``ps_ttm`` 列

        Returns
        -------
        pd.Series
            SP（1/PS）或 PS 值
        """
        ps_col = self._find_ps_column(data)
        ps = data[ps_col].copy()

        if self.inverse:
            sp = pd.Series(index=ps.index, dtype=float)
            valid_mask = ps > 0
            sp[valid_mask] = 1.0 / ps[valid_mask]
            return sp
        return ps

    @staticmethod
    def _find_ps_column(data: pd.DataFrame) -> str:
        """查找 PS 列名"""
        for col in ["ps_ttm", "ps", "PS_TTM", "PS"]:
            if col in data.columns:
                return col
        raise KeyError(
            f"DataFrame 中未找到 PS 列，可用列: {list(data.columns)}"
        )

    def get_params(self) -> dict:
        return {"inverse": self.inverse}
