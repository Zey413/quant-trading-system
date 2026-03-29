"""数据管理器 - 统一数据获取入口，带 Parquet 文件缓存

DataManager 是数据层的门面(Facade)，封装了:
- 数据源选择与切换
- Parquet 文件缓存（避免重复API调用）
- 列名标准化
- 数据验证
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from quant_trading.core.config import AppConfig
from quant_trading.data.base import DataSource, DataSourceRegistry

logger = logging.getLogger(__name__)

# 标准化列名集合
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["amount"]


class DataManager:
    """数据管理器 - 门面模式

    提供统一的数据获取接口，自动管理缓存和数据源切换。

    Parameters
    ----------
    config : AppConfig
        应用全局配置，包含数据源配置、缓存目录等

    Examples
    --------
    >>> from quant_trading.core.config import AppConfig
    >>> config = AppConfig()
    >>> dm = DataManager(config)
    >>> df = dm.fetch_daily("000001", "20230101", "20231231")
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ds_config = config.data_source

        # 初始化缓存目录
        self._cache_dir = Path(self._ds_config.cache_dir)
        if self._ds_config.cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 延迟初始化数据源（首次使用时创建）
        self._source: DataSource | None = None

    @property
    def source(self) -> DataSource:
        """获取当前数据源实例（延迟初始化）"""
        if self._source is None:
            self._source = DataSourceRegistry.get(self._ds_config.default_source)
            logger.info("Initialized data source: %s", self._source.get_name())
        return self._source

    @property
    def source_name(self) -> str:
        """当前数据源名称"""
        return self.source.get_name()

    def set_source(self, name: str) -> None:
        """切换数据源

        Parameters
        ----------
        name : str
            已注册的数据源名称
        """
        self._source = DataSourceRegistry.get(name)
        logger.info("Switched data source to: %s", name)

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        source_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> Path:
        """生成缓存文件路径

        格式: {cache_dir}/{source}_{symbol}_{start}_{end}_{adjust}.parquet
        """
        # 统一日期格式：去除分隔符
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        filename = f"{source_name}_{symbol}_{start}_{end}_{adjust}.parquet"
        return self._cache_dir / filename

    def _read_cache(self, cache_path: Path) -> pd.DataFrame | None:
        """读取缓存文件，不存在或损坏则返回 None"""
        if not self._ds_config.cache_enabled:
            return None
        if not cache_path.exists():
            return None
        try:
            df = pd.read_parquet(cache_path)
            logger.debug("Cache hit: %s", cache_path.name)
            return df
        except Exception:
            logger.warning("Corrupted cache file, removing: %s", cache_path.name)
            cache_path.unlink(missing_ok=True)
            return None

    def _write_cache(self, cache_path: Path, df: pd.DataFrame) -> None:
        """写入缓存文件"""
        if not self._ds_config.cache_enabled:
            return
        if df.empty:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False, engine="pyarrow")
            logger.debug("Cache written: %s", cache_path.name)
        except Exception:
            logger.warning("Failed to write cache: %s", cache_path.name, exc_info=True)

    # ------------------------------------------------------------------
    # 数据标准化
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """标准化 DataFrame 列名和数据类型

        确保输出包含标准列: date, open, high, low, close, volume
        可选列: amount

        数据按 date 升序排列，date 列为 datetime 类型。
        """
        if df.empty:
            return df

        # 只保留标准列 + 可选列
        keep_cols = [c for c in STANDARD_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]
        df = df[keep_cols].copy()

        # 确保必需列存在
        missing = set(STANDARD_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns after standardization: {missing}")

        # 类型转换
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        # 按日期升序排列
        df = df.sort_values("date").reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线数据，优先使用缓存

        Parameters
        ----------
        symbol : str
            股票代码（纯数字），例如 "000001", "600519"
        start_date : str
            开始日期 "YYYYMMDD" 或 "YYYY-MM-DD"
        end_date : str
            结束日期 "YYYYMMDD" 或 "YYYY-MM-DD"
        adjust : str
            复权类型: "qfq", "hfq", "none"

        Returns
        -------
        pd.DataFrame
            标准化列: date, open, high, low, close, volume[, amount]
        """
        src = self.source
        cache_path = self._cache_key(src.get_name(), symbol, start_date, end_date, adjust)

        # 1. 尝试读缓存
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        # 2. 从数据源获取
        logger.info(
            "Fetching daily data: %s %s %s~%s adjust=%s",
            src.get_name(),
            symbol,
            start_date,
            end_date,
            adjust,
        )
        df = src.fetch_daily(symbol, start_date, end_date, adjust=adjust)

        # 3. 标准化
        df = self._standardize(df)

        # 4. 写缓存
        self._write_cache(cache_path, df)

        return df

    def fetch_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表

        Returns
        -------
        pd.DataFrame
            至少包含 code, name 列
        """
        logger.info("Fetching stock list from: %s", self.source.get_name())
        return self.source.fetch_stock_list()

    def fetch_realtime(self, symbol: str) -> dict:
        """获取实时行情

        Parameters
        ----------
        symbol : str
            股票代码

        Returns
        -------
        dict
            实时行情字段
        """
        return self.source.fetch_realtime(symbol)

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def clear_cache(self, symbol: str | None = None) -> int:
        """清除缓存

        Parameters
        ----------
        symbol : str, optional
            指定股票代码清除该股票的缓存，不指定则清除全部

        Returns
        -------
        int
            清除的文件数量
        """
        if not self._cache_dir.exists():
            return 0

        if symbol is not None:
            # 清除指定股票的缓存
            pattern = f"*_{symbol}_*.parquet"
            files = list(self._cache_dir.glob(pattern))
            for f in files:
                f.unlink(missing_ok=True)
            count = len(files)
            logger.info("Cleared %d cache files for symbol %s", count, symbol)
            return count

        # 清除全部缓存
        files = list(self._cache_dir.glob("*.parquet"))
        for f in files:
            f.unlink(missing_ok=True)
        count = len(files)
        logger.info("Cleared all %d cache files", count)
        return count

    def cache_info(self) -> dict:
        """获取缓存统计信息

        Returns
        -------
        dict
            包含 enabled, directory, file_count, total_size_mb
        """
        if not self._cache_dir.exists():
            return {
                "enabled": self._ds_config.cache_enabled,
                "directory": str(self._cache_dir),
                "file_count": 0,
                "total_size_mb": 0.0,
            }

        files = list(self._cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "enabled": self._ds_config.cache_enabled,
            "directory": str(self._cache_dir),
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
