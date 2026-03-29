"""特征工程模块

提供 FeatureEngineer 类，用于从OHLCV行情数据中构建机器学习所需的特征矩阵。
支持技术指标特征、价格特征、成交量特征、时间特征、滞后特征、滚动特征、
标签生成以及特征选择。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quant_trading.indicators.momentum import CCI, ROC, WilliamsR
from quant_trading.indicators.oscillator import KDJ, RSI
from quant_trading.indicators.trend import EMA, MACD, SMA
from quant_trading.indicators.volatility import ATR, BollingerBands
from quant_trading.indicators.volume import OBV, VWAP


class FeatureEngineer:
    """特征工程器，从OHLCV数据中提取ML特征。

    该类提供一系列方法，用于从原始行情数据中构建丰富的特征矩阵。
    所有方法都接收并返回 DataFrame，方便链式调用。

    Example
    -------
    >>> fe = FeatureEngineer()
    >>> X, y, feature_names = fe.build_feature_matrix(df)
    """

    def __init__(
        self,
        sma_windows: list[int] | None = None,
        ema_windows: list[int] | None = None,
        rsi_periods: list[int] | None = None,
        atr_periods: list[int] | None = None,
        roc_periods: list[int] | None = None,
        lag_periods: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        label_method: str = "direction",
        label_threshold: float = 0.0,
        label_horizon: int = 1,
    ) -> None:
        """初始化特征工程器。

        Parameters
        ----------
        sma_windows : list[int], optional
            SMA周期列表，默认 [5, 10, 20, 60]。
        ema_windows : list[int], optional
            EMA周期列表，默认 [12, 26]。
        rsi_periods : list[int], optional
            RSI周期列表，默认 [14]。
        atr_periods : list[int], optional
            ATR周期列表，默认 [14]。
        roc_periods : list[int], optional
            ROC周期列表，默认 [12]。
        lag_periods : list[int], optional
            滞后期数列表，默认 [1, 2, 3, 5, 10]。
        rolling_windows : list[int], optional
            滚动窗口列表，默认 [5, 10, 20]。
        label_method : str
            标签生成方法: 'return', 'direction', 'triple_barrier'。
        label_threshold : float
            标签阈值（用于 direction 和 triple_barrier）。
        label_horizon : int
            标签前瞻周期数。
        """
        self.sma_windows = sma_windows or [5, 10, 20, 60]
        self.ema_windows = ema_windows or [12, 26]
        self.rsi_periods = rsi_periods or [14]
        self.atr_periods = atr_periods or [14]
        self.roc_periods = roc_periods or [12]
        self.lag_periods = lag_periods or [1, 2, 3, 5, 10]
        self.rolling_windows = rolling_windows or [5, 10, 20]
        self.label_method = label_method
        self.label_threshold = label_threshold
        self.label_horizon = label_horizon
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        """返回最近一次 build_feature_matrix 生成的特征名列表。"""
        return self._feature_names.copy()

    # ------------------------------------------------------------------
    # 技术指标特征
    # ------------------------------------------------------------------

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用现有指标库计算技术指标特征。

        添加的特征包括: SMA、EMA、RSI、MACD、布林带、ATR、OBV、VWAP、
        ROC、Williams %R、CCI、KDJ。

        Parameters
        ----------
        df : pd.DataFrame
            至少包含 open, high, low, close, volume 列。

        Returns
        -------
        pd.DataFrame
            添加了技术指标列的 DataFrame。
        """
        df = df.copy()

        # SMA
        for w in self.sma_windows:
            col = f"sma_{w}"
            if col not in df.columns:
                df = SMA(window=w).calculate(df)

        # EMA
        for w in self.ema_windows:
            col = f"ema_{w}"
            if col not in df.columns:
                df = EMA(window=w).calculate(df)

        # RSI
        for p in self.rsi_periods:
            col = f"rsi_{p}"
            if col not in df.columns:
                df = RSI(period=p).calculate(df)

        # MACD
        if "macd" not in df.columns:
            df = MACD().calculate(df)

        # BollingerBands
        if "bb_upper" not in df.columns:
            df = BollingerBands(window=20).calculate(df)

        # ATR
        for p in self.atr_periods:
            col = f"atr_{p}"
            if col not in df.columns:
                df = ATR(period=p).calculate(df)

        # OBV
        if "obv" not in df.columns:
            df = OBV().calculate(df)

        # VWAP
        if "vwap" not in df.columns:
            df = VWAP().calculate(df)

        # ROC
        for p in self.roc_periods:
            col = f"roc_{p}"
            if col not in df.columns:
                df = ROC(period=p).calculate(df)

        # Williams %R
        if "williams_r_14" not in df.columns:
            df = WilliamsR(period=14).calculate(df)

        # CCI
        if "cci_20" not in df.columns:
            df = CCI(period=20).calculate(df)

        # KDJ
        if "k" not in df.columns:
            df = KDJ().calculate(df)

        return df

    # ------------------------------------------------------------------
    # 价格特征
    # ------------------------------------------------------------------

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格相关特征。

        包括: 日收益率、对数收益率、多周期收益率、日内波动率、
        价格相对位置、价格比率等。

        Parameters
        ----------
        df : pd.DataFrame
            至少包含 open, high, low, close 列。

        Returns
        -------
        pd.DataFrame
            添加了价格特征列的 DataFrame。
        """
        df = df.copy()
        close = df["close"]

        # 日收益率
        df["return_1d"] = close.pct_change(1)

        # 对数收益率
        df["log_return_1d"] = np.log(close / close.shift(1))

        # 多周期收益率
        for n in [2, 3, 5, 10, 20]:
            df[f"return_{n}d"] = close.pct_change(n)

        # 日内波动率 (high-low range / close)
        df["intraday_range"] = (df["high"] - df["low"]) / close

        # 日内波动方向 (close - open) / open
        df["intraday_return"] = (close - df["open"]) / df["open"]

        # 上影线比例
        df["upper_shadow"] = (df["high"] - np.maximum(close, df["open"])) / (
            df["high"] - df["low"] + 1e-10
        )

        # 下影线比例
        df["lower_shadow"] = (np.minimum(close, df["open"]) - df["low"]) / (
            df["high"] - df["low"] + 1e-10
        )

        # 历史波动率 (5日、10日、20日)
        for w in [5, 10, 20]:
            df[f"volatility_{w}d"] = df["return_1d"].rolling(window=w).std()

        # 价格相对SMA位置 (价格偏离度)
        for w in self.sma_windows:
            sma_col = f"sma_{w}"
            if sma_col in df.columns:
                df[f"price_sma_{w}_ratio"] = close / df[sma_col] - 1.0

        # 价格在N日高低区间中的相对位置
        for w in [5, 10, 20]:
            high_w = df["high"].rolling(window=w).max()
            low_w = df["low"].rolling(window=w).min()
            denom = high_w - low_w
            df[f"price_position_{w}d"] = np.where(
                denom != 0, (close - low_w) / denom, 0.5
            )

        # 缺口 (gap): open相对前日close的跳空
        df["gap"] = df["open"] / close.shift(1) - 1.0

        return df

    # ------------------------------------------------------------------
    # 成交量特征
    # ------------------------------------------------------------------

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量相关特征。

        包括: 量比（相对N日均量）、成交量变化率、成交额、
        价量相关性等。

        Parameters
        ----------
        df : pd.DataFrame
            至少包含 close, volume 列。

        Returns
        -------
        pd.DataFrame
            添加了成交量特征列的 DataFrame。
        """
        df = df.copy()
        volume = df["volume"]

        # 量比 (当前成交量 / N日平均成交量)
        for w in [5, 10, 20]:
            avg_vol = volume.rolling(window=w).mean()
            df[f"volume_ratio_{w}d"] = np.where(
                avg_vol != 0, volume / avg_vol, 1.0
            )

        # 成交量变化率
        df["volume_change"] = volume.pct_change(1)

        # 成交量的对数值 (消除量纲差异)
        df["log_volume"] = np.log(volume + 1)

        # 成交额 (如果有close和volume)
        df["turnover"] = df["close"] * volume

        # 成交额均值比
        turnover = df["turnover"]
        for w in [5, 10]:
            avg_to = turnover.rolling(window=w).mean()
            df[f"turnover_ratio_{w}d"] = np.where(
                avg_to != 0, turnover / avg_to, 1.0
            )

        # 量价相关性 (N日滚动)
        for w in [10, 20]:
            df[f"vol_price_corr_{w}d"] = (
                volume.rolling(window=w).corr(df["close"])
            )

        # OBV变化率
        if "obv" in df.columns:
            df["obv_change"] = df["obv"].pct_change(1)
            df["obv_change"] = df["obv_change"].replace(
                [np.inf, -np.inf], 0.0
            )

        return df

    # ------------------------------------------------------------------
    # 时间特征
    # ------------------------------------------------------------------

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间相关特征。

        如果 DataFrame 的 index 是 DatetimeIndex，则提取月份、星期、
        季度、是否月初/月末等时间特征。

        Parameters
        ----------
        df : pd.DataFrame
            索引为 DatetimeIndex 或包含 date 列。

        Returns
        -------
        pd.DataFrame
            添加了时间特征列的 DataFrame。
        """
        df = df.copy()

        # 尝试获取datetime信息
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        elif "date" in df.columns:
            dt = pd.to_datetime(df["date"])
        else:
            # 无法提取时间特征，直接返回
            return df

        # 月份 (1-12)
        df["month"] = dt.month

        # 星期几 (0=Monday, 6=Sunday)
        df["day_of_week"] = dt.dayofweek

        # 季度 (1-4)
        df["quarter"] = dt.quarter

        # 月中的第几天
        df["day_of_month"] = dt.day

        # 是否月初 (前5个交易日)
        df["is_month_start"] = (dt.day <= 5).astype(int)

        # 是否月末 (后5个交易日)
        df["is_month_end"] = (dt.day >= 25).astype(int)

        # 正弦/余弦编码月份 (捕获周期性)
        df["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

        # 正弦/余弦编码星期
        df["dow_sin"] = np.sin(2 * np.pi * dt.dayofweek / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dayofweek / 5)

        return df

    # ------------------------------------------------------------------
    # 滞后特征
    # ------------------------------------------------------------------

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        lags: list[int] | None = None,
    ) -> pd.DataFrame:
        """添加滞后特征 (lag features)。

        将指定列的历史值作为新特征。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。
        columns : list[str], optional
            需要创建滞后特征的列名，默认 ['close', 'volume', 'return_1d']。
        lags : list[int], optional
            滞后期数列表，默认使用 self.lag_periods。

        Returns
        -------
        pd.DataFrame
            添加了滞后特征的 DataFrame。
        """
        df = df.copy()
        if columns is None:
            columns = [c for c in ["close", "volume", "return_1d"] if c in df.columns]
        if lags is None:
            lags = self.lag_periods

        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    # ------------------------------------------------------------------
    # 滚动统计特征
    # ------------------------------------------------------------------

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """添加滚动统计特征。

        对指定列计算滚动均值、标准差、最大值、最小值、偏度。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。
        columns : list[str], optional
            需要计算滚动特征的列名，默认 ['return_1d']。
        windows : list[int], optional
            滚动窗口列表，默认使用 self.rolling_windows。

        Returns
        -------
        pd.DataFrame
            添加了滚动统计特征的 DataFrame。
        """
        df = df.copy()
        if columns is None:
            columns = [c for c in ["return_1d"] if c in df.columns]
        if windows is None:
            windows = self.rolling_windows

        for col in columns:
            if col not in df.columns:
                continue
            for w in windows:
                rolling = df[col].rolling(window=w, min_periods=w)
                df[f"{col}_rmean_{w}"] = rolling.mean()
                df[f"{col}_rstd_{w}"] = rolling.std()
                df[f"{col}_rmax_{w}"] = rolling.max()
                df[f"{col}_rmin_{w}"] = rolling.min()
                df[f"{col}_rskew_{w}"] = rolling.skew()

        return df

    # ------------------------------------------------------------------
    # 标签生成
    # ------------------------------------------------------------------

    def create_labels(
        self,
        df: pd.DataFrame,
        method: Literal["return", "direction", "triple_barrier"] | None = None,
        threshold: float | None = None,
        horizon: int | None = None,
    ) -> pd.Series:
        """生成ML训练标签。

        Parameters
        ----------
        df : pd.DataFrame
            包含 close 列的行情数据。
        method : str, optional
            标签方法，默认使用 self.label_method:
            - 'return': 未来N期收益率（回归标签）
            - 'direction': 二分类（1=涨, 0=跌）
            - 'triple_barrier': 三分类（1=涨超阈值, -1=跌超阈值, 0=不变）
        threshold : float, optional
            方向/三屏障阈值，默认使用 self.label_threshold。
        horizon : int, optional
            前瞻周期数，默认使用 self.label_horizon。

        Returns
        -------
        pd.Series
            标签序列，与 df 的 index 对齐。
        """
        method = method or self.label_method
        threshold = threshold if threshold is not None else self.label_threshold
        horizon = horizon or self.label_horizon

        close = df["close"]
        future_return = close.shift(-horizon) / close - 1.0

        if method == "return":
            return future_return

        elif method == "direction":
            labels = (future_return > threshold).astype(int)
            return labels

        elif method == "triple_barrier":
            labels = pd.Series(0, index=df.index)
            labels[future_return > threshold] = 1
            labels[future_return < -threshold] = -1
            return labels

        else:
            raise ValueError(
                f"不支持的标签方法: {method!r}。"
                f"可选: 'return', 'direction', 'triple_barrier'"
            )

    # ------------------------------------------------------------------
    # 特征选择
    # ------------------------------------------------------------------

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: Literal["mutual_info", "correlation", "variance"] = "mutual_info",
        k: int = 20,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[int], list[str]]:
        """特征选择，返回筛选后的特征矩阵。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵，shape (n_samples, n_features)。
        y : np.ndarray
            标签向量，shape (n_samples,)。
        method : str
            选择方法:
            - 'mutual_info': 基于互信息的特征选择（离散化）
            - 'correlation': 基于与目标相关性的选择
            - 'variance': 基于方差的过滤
        k : int
            保留的特征数量。
        feature_names : list[str], optional
            特征名列表。

        Returns
        -------
        tuple[np.ndarray, list[int], list[str]]
            (筛选后的X, 选中特征的索引, 选中特征的名称)
        """
        n_features = X.shape[1]
        k = min(k, n_features)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if method == "mutual_info":
            scores = self._mutual_info_scores(X, y)
        elif method == "correlation":
            scores = self._correlation_scores(X, y)
        elif method == "variance":
            scores = np.nanvar(X, axis=0)
        else:
            raise ValueError(
                f"不支持的特征选择方法: {method!r}。"
                f"可选: 'mutual_info', 'correlation', 'variance'"
            )

        # 选取得分最高的k个特征
        top_indices = np.argsort(scores)[::-1][:k].tolist()
        top_indices.sort()  # 保持原始顺序
        selected_names = [feature_names[i] for i in top_indices]

        return X[:, top_indices], top_indices, selected_names

    def _mutual_info_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算每个特征与目标之间的互信息（离散化近似）。

        使用等频分箱将连续特征离散化，然后计算互信息。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵。
        y : np.ndarray
            目标向量。

        Returns
        -------
        np.ndarray
            每个特征的互信息得分。
        """
        n_samples, n_features = X.shape
        n_bins = min(10, max(2, int(np.sqrt(n_samples))))
        scores = np.zeros(n_features)

        # 离散化y
        y_discrete = self._discretize(y, n_bins)

        for j in range(n_features):
            x_col = X[:, j]
            # 跳过全NaN或常数列
            valid = ~np.isnan(x_col)
            if valid.sum() < 2 or np.nanstd(x_col) < 1e-12:
                scores[j] = 0.0
                continue

            x_discrete = self._discretize(x_col[valid], n_bins)
            y_valid = y_discrete[valid]
            scores[j] = self._compute_mi(x_discrete, y_valid)

        return scores

    @staticmethod
    def _discretize(arr: np.ndarray, n_bins: int) -> np.ndarray:
        """将连续数组等频离散化为整数标签。"""
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        edges = np.percentile(arr[~np.isnan(arr)], percentiles)
        return np.digitize(arr, edges)

    @staticmethod
    def _compute_mi(x: np.ndarray, y: np.ndarray) -> float:
        """计算两个离散数组之间的互信息。"""
        n = len(x)
        if n == 0:
            return 0.0

        # 联合分布
        xy_pairs = np.column_stack([x, y])
        # 使用字典计数
        joint_counts: dict[tuple, int] = {}
        x_counts: dict[int, int] = {}
        y_counts: dict[int, int] = {}

        for i in range(n):
            xi, yi = int(xy_pairs[i, 0]), int(xy_pairs[i, 1])
            key = (xi, yi)
            joint_counts[key] = joint_counts.get(key, 0) + 1
            x_counts[xi] = x_counts.get(xi, 0) + 1
            y_counts[yi] = y_counts.get(yi, 0) + 1

        mi = 0.0
        for (xi, yi), n_xy in joint_counts.items():
            p_xy = n_xy / n
            p_x = x_counts[xi] / n
            p_y = y_counts[yi] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

        return max(0.0, mi)

    def _correlation_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算每个特征与目标之间的绝对相关系数。"""
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        for j in range(n_features):
            x_col = X[:, j]
            valid = ~(np.isnan(x_col) | np.isnan(y))
            if valid.sum() < 2:
                scores[j] = 0.0
                continue
            xv = x_col[valid]
            yv = y[valid]
            if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                scores[j] = 0.0
                continue
            corr = np.corrcoef(xv, yv)[0, 1]
            scores[j] = abs(corr) if not np.isnan(corr) else 0.0

        return scores

    # ------------------------------------------------------------------
    # 一键特征构建
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_price: bool = True,
        include_volume: bool = True,
        include_time: bool = True,
        include_lag: bool = True,
        include_rolling: bool = True,
        dropna: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """一键构建完整特征矩阵和标签。

        按顺序调用所有特征生成方法，生成标签，去除NaN，返回
        可直接用于模型训练的numpy数组。

        Parameters
        ----------
        df : pd.DataFrame
            原始OHLCV行情数据。
        include_technical : bool
            是否包含技术指标特征。
        include_price : bool
            是否包含价格特征。
        include_volume : bool
            是否包含成交量特征。
        include_time : bool
            是否包含时间特征。
        include_lag : bool
            是否包含滞后特征。
        include_rolling : bool
            是否包含滚动统计特征。
        dropna : bool
            是否去除包含NaN的行。

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            (特征矩阵X, 标签向量y, 特征名列表)
        """
        result = df.copy()
        original_cols = set(result.columns)

        # 依次添加各类特征
        if include_technical:
            result = self.add_technical_features(result)

        if include_price:
            result = self.add_price_features(result)

        if include_volume:
            result = self.add_volume_features(result)

        if include_time:
            result = self.add_time_features(result)

        if include_lag:
            result = self.add_lag_features(result)

        if include_rolling:
            result = self.add_rolling_features(result)

        # 生成标签
        labels = self.create_labels(result)
        result["_label"] = labels

        # 提取特征列 (排除原始OHLCV列和标签列)
        exclude_cols = original_cols | {"_label", "signal"}
        feature_cols = [c for c in result.columns if c not in exclude_cols]
        self._feature_names = feature_cols

        if dropna:
            valid_mask = result[feature_cols + ["_label"]].notna().all(axis=1)
            result = result[valid_mask]

        X = result[feature_cols].values.astype(np.float64)
        y = result["_label"].values.astype(np.float64)

        # 处理可能残留的inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, feature_cols
