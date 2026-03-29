"""指标工具函数

提供一键添加指标、指标相关性分析等实用工具。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.momentum import CCI, ROC, WilliamsR
from quant_trading.indicators.oscillator import RSI
from quant_trading.indicators.trend import EMA, MACD, SMA
from quant_trading.indicators.volatility import ATR, BollingerBands
from quant_trading.indicators.volume import OBV, VWAP


# 默认指标配置
_DEFAULT_CONFIG: dict = {
    "sma": [5, 10, 20, 60],
    "ema": [12, 26],
    "rsi": [14],
    "macd": True,
    "bollinger": [20],
    "atr": [14],
    "obv": True,
    "vwap": True,
    "roc": [12],
    "williams_r": [14],
    "cci": [20],
}


def add_all_indicators(
    data: pd.DataFrame, config: dict | None = None
) -> pd.DataFrame:
    """一键添加所有常用指标

    默认添加:
        SMA(5,10,20,60), EMA(12,26), RSI(14), MACD, BB(20), ATR(14),
        OBV, VWAP, ROC(12), WilliamsR(14), CCI(20)

    可通过 config 自定义要添加的指标及参数。

    Parameters
    ----------
    data : pd.DataFrame
        至少包含 open/high/low/close/volume 的行情数据。
    config : dict, optional
        指标配置字典。键为指标类型名，值为参数列表或 True/False。
        例::

            {
                "sma": [5, 20],       # 添加 SMA(5) 和 SMA(20)
                "ema": [12, 26],      # 添加 EMA(12) 和 EMA(26)
                "rsi": [14],          # 添加 RSI(14)
                "macd": True,         # 添加 MACD (默认参数)
                "bollinger": [20],    # 添加 BB(20)
                "atr": [14],          # 添加 ATR(14)
                "obv": True,          # 添加 OBV
                "vwap": True,         # 添加 VWAP
                "roc": [12],          # 添加 ROC(12)
                "williams_r": [14],   # 添加 WilliamsR(14)
                "cci": [20],          # 添加 CCI(20)
            }

        设置为 ``False`` 或空列表可跳过某类指标::

            {"sma": False, "vwap": False}  # 不添加 SMA 和 VWAP

    Returns
    -------
    pd.DataFrame
        添加了指标列的 DataFrame（原地修改并返回）。
    """
    cfg = _DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)

    # SMA
    sma_windows = cfg.get("sma", [])
    if sma_windows:
        for w in sma_windows:
            SMA(window=w).calculate(data)

    # EMA
    ema_windows = cfg.get("ema", [])
    if ema_windows:
        for w in ema_windows:
            EMA(window=w).calculate(data)

    # RSI
    rsi_periods = cfg.get("rsi", [])
    if rsi_periods:
        for p in rsi_periods:
            RSI(period=p).calculate(data)

    # MACD
    if cfg.get("macd"):
        MACD().calculate(data)

    # BollingerBands
    bb_windows = cfg.get("bollinger", [])
    if bb_windows:
        for w in bb_windows:
            BollingerBands(window=w).calculate(data)

    # ATR
    atr_periods = cfg.get("atr", [])
    if atr_periods:
        for p in atr_periods:
            ATR(period=p).calculate(data)

    # OBV
    if cfg.get("obv"):
        OBV().calculate(data)

    # VWAP
    if cfg.get("vwap"):
        VWAP().calculate(data)

    # ROC
    roc_periods = cfg.get("roc", [])
    if roc_periods:
        for p in roc_periods:
            ROC(period=p).calculate(data)

    # WilliamsR
    wr_periods = cfg.get("williams_r", [])
    if wr_periods:
        for p in wr_periods:
            WilliamsR(period=p).calculate(data)

    # CCI
    cci_periods = cfg.get("cci", [])
    if cci_periods:
        for p in cci_periods:
            CCI(period=p).calculate(data)

    return data


def calculate_indicator_correlation(
    data: pd.DataFrame, indicators: list[str]
) -> pd.DataFrame:
    """计算指标间的相关性矩阵

    Parameters
    ----------
    data : pd.DataFrame
        包含指标列的 DataFrame。
    indicators : list[str]
        要计算相关性的指标列名列表。

    Returns
    -------
    pd.DataFrame
        相关性矩阵 (Pearson 相关系数)。

    Raises
    ------
    ValueError
        当指定的列不存在于 data 中时抛出。
    """
    missing = [col for col in indicators if col not in data.columns]
    if missing:
        raise ValueError(
            f"以下指标列不存在于数据中: {missing}。"
            f"现有列: {list(data.columns)}"
        )

    subset = data[indicators].dropna()
    if subset.empty:
        return pd.DataFrame(
            index=indicators, columns=indicators, dtype=float
        )

    return subset.corr()
