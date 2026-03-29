"""FeatureEngineer 单元测试

至少15个测试用例，覆盖技术指标特征、价格特征、成交量特征、时间特征、
滞后特征、滚动特征、标签生成和特征选择。所有测试数据使用numpy生成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading.ml.features import FeatureEngineer


# ======================================================================
# 辅助函数
# ======================================================================


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成模拟OHLCV数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ======================================================================
# 测试用例
# ======================================================================


class TestAddTechnicalFeatures:
    """技术指标特征添加。"""

    def test_adds_sma_columns(self):
        """应添加SMA列。"""
        fe = FeatureEngineer(sma_windows=[5, 10])
        df = _make_ohlcv()
        result = fe.add_technical_features(df)
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns

    def test_adds_rsi_column(self):
        """应添加RSI列。"""
        fe = FeatureEngineer(rsi_periods=[14])
        df = _make_ohlcv()
        result = fe.add_technical_features(df)
        assert "rsi_14" in result.columns

    def test_adds_macd_columns(self):
        """应添加MACD相关列。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_technical_features(df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_adds_bollinger_columns(self):
        """应添加布林带列。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_technical_features(df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns

    def test_adds_kdj_columns(self):
        """应添加KDJ列。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_technical_features(df)
        assert "k" in result.columns
        assert "d" in result.columns
        assert "j" in result.columns


class TestAddPriceFeatures:
    """价格特征添加。"""

    def test_adds_return_columns(self):
        """应添加收益率列。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_price_features(df)
        assert "return_1d" in result.columns
        assert "log_return_1d" in result.columns

    def test_adds_volatility_columns(self):
        """应添加波动率列。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_price_features(df)
        assert "volatility_5d" in result.columns
        assert "volatility_20d" in result.columns

    def test_adds_intraday_features(self):
        """应添加日内特征。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_price_features(df)
        assert "intraday_range" in result.columns
        assert "intraday_return" in result.columns
        assert "upper_shadow" in result.columns
        assert "lower_shadow" in result.columns

    def test_adds_gap_feature(self):
        """应添加跳空缺口特征。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_price_features(df)
        assert "gap" in result.columns


class TestAddVolumeFeatures:
    """成交量特征添加。"""

    def test_adds_volume_ratio(self):
        """应添加量比列。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_volume_features(df)
        assert "volume_ratio_5d" in result.columns
        assert "volume_ratio_10d" in result.columns

    def test_adds_log_volume(self):
        """应添加对数成交量。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_volume_features(df)
        assert "log_volume" in result.columns


class TestAddTimeFeatures:
    """时间特征添加。"""

    def test_adds_time_features_with_datetime_index(self):
        """DatetimeIndex时应添加时间特征。"""
        fe = FeatureEngineer()
        df = _make_ohlcv()
        result = fe.add_time_features(df)
        assert "month" in result.columns
        assert "day_of_week" in result.columns
        assert "quarter" in result.columns
        assert "month_sin" in result.columns

    def test_no_time_features_without_datetime(self):
        """非DatetimeIndex且无date列时应跳过。"""
        fe = FeatureEngineer()
        df = _make_ohlcv().reset_index(drop=True)
        result = fe.add_time_features(df)
        assert "month" not in result.columns


class TestLagFeatures:
    """滞后特征添加。"""

    def test_adds_lag_columns(self):
        """应添加滞后列。"""
        fe = FeatureEngineer(lag_periods=[1, 2, 3])
        df = _make_ohlcv()
        result = fe.add_lag_features(df, columns=["close"], lags=[1, 2])
        assert "close_lag_1" in result.columns
        assert "close_lag_2" in result.columns

    def test_lag_values_correct(self):
        """滞后值应正确。"""
        fe = FeatureEngineer()
        df = _make_ohlcv(n=10)
        result = fe.add_lag_features(df, columns=["close"], lags=[1])
        # lag_1应等于前一行的close
        assert result["close_lag_1"].iloc[1] == result["close"].iloc[0]


class TestRollingFeatures:
    """滚动统计特征。"""

    def test_adds_rolling_stats(self):
        """应添加滚动均值/标准差/极值。"""
        fe = FeatureEngineer(rolling_windows=[5])
        df = _make_ohlcv()
        df["return_1d"] = df["close"].pct_change()
        result = fe.add_rolling_features(df, columns=["return_1d"], windows=[5])
        assert "return_1d_rmean_5" in result.columns
        assert "return_1d_rstd_5" in result.columns
        assert "return_1d_rmax_5" in result.columns
        assert "return_1d_rmin_5" in result.columns


class TestCreateLabels:
    """标签生成。"""

    def test_return_label(self):
        """return方法应返回未来收益率。"""
        fe = FeatureEngineer()
        df = _make_ohlcv(n=50)
        labels = fe.create_labels(df, method="return", horizon=1)
        assert len(labels) == len(df)
        # 最后一行应为NaN (无未来数据)
        assert np.isnan(labels.iloc[-1])

    def test_direction_label(self):
        """direction方法应返回0/1。"""
        fe = FeatureEngineer()
        df = _make_ohlcv(n=50)
        labels = fe.create_labels(df, method="direction", horizon=1)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_triple_barrier_label(self):
        """triple_barrier方法应返回-1/0/1。"""
        fe = FeatureEngineer()
        df = _make_ohlcv(n=100)
        labels = fe.create_labels(
            df, method="triple_barrier", threshold=0.005, horizon=1
        )
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_invalid_label_method(self):
        """非法方法应抛出ValueError。"""
        fe = FeatureEngineer()
        df = _make_ohlcv(n=10)
        with pytest.raises(ValueError, match="不支持的标签方法"):
            fe.create_labels(df, method="invalid")


class TestSelectFeatures:
    """特征选择。"""

    def test_mutual_info_selection(self):
        """互信息选择应返回指定数量的特征。"""
        fe = FeatureEngineer()
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        X_selected, indices, names = fe.select_features(
            X, y, method="mutual_info", k=5
        )
        assert X_selected.shape[1] == 5
        assert len(indices) == 5
        assert len(names) == 5

    def test_correlation_selection(self):
        """相关性选择应正常工作。"""
        fe = FeatureEngineer()
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = X[:, 0] * 3 + rng.randn(100) * 0.1

        X_selected, indices, names = fe.select_features(
            X, y, method="correlation", k=3
        )
        assert X_selected.shape[1] == 3
        # 第0个特征与y高度相关，应被选中
        assert 0 in indices

    def test_variance_selection(self):
        """方差选择应正常工作。"""
        fe = FeatureEngineer()
        rng = np.random.RandomState(42)
        # 第0列方差大，第9列几乎为常数
        X = rng.randn(100, 10)
        X[:, 9] = 1.0  # 常数列
        y = rng.randint(0, 2, 100)

        X_selected, indices, names = fe.select_features(
            X, y, method="variance", k=5
        )
        assert X_selected.shape[1] == 5
        assert 9 not in indices  # 常数列不应被选中


class TestBuildFeatureMatrix:
    """一键特征构建。"""

    def test_build_returns_valid_matrix(self):
        """应返回无NaN的特征矩阵。"""
        fe = FeatureEngineer(
            sma_windows=[5, 10],
            ema_windows=[12],
            rsi_periods=[14],
            atr_periods=[14],
            roc_periods=[12],
            lag_periods=[1, 2],
            rolling_windows=[5],
        )
        df = _make_ohlcv(n=200)
        X, y, feature_names = fe.build_feature_matrix(df)

        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0
        assert len(feature_names) == X.shape[1]
        # 无NaN
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_feature_names_stored(self):
        """特征名应被存储。"""
        fe = FeatureEngineer(
            sma_windows=[5],
            ema_windows=[12],
            lag_periods=[1],
            rolling_windows=[5],
        )
        df = _make_ohlcv(n=100)
        X, y, feature_names = fe.build_feature_matrix(df)
        assert fe.feature_names == feature_names
