"""数据源模块单元测试

测试覆盖:
- DataSourceRegistry 注册/获取/列表
- DataManager 缓存读写、标准化、清除
- AKShare 数据源列名映射
- Tushare 股票代码转换
- 不依赖真实 API 调用（全部 mock）
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_trading.core.config import AppConfig, DataSourceConfig
from quant_trading.data.base import DataSource, DataSourceRegistry
from quant_trading.data.manager import DataManager


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _clean_registry():
    """每个测试前后保存/恢复注册表状态，防止测试间互相影响"""
    original = DataSourceRegistry._sources.copy()
    yield
    DataSourceRegistry._sources = original


@pytest.fixture()
def sample_akshare_df() -> pd.DataFrame:
    """模拟 AKShare stock_zh_a_hist 返回的中文列名 DataFrame"""
    return pd.DataFrame(
        {
            "日期": ["2023-01-03", "2023-01-04", "2023-01-05"],
            "开盘": [13.50, 13.62, 13.80],
            "收盘": [13.60, 13.78, 13.72],
            "最高": [13.70, 13.85, 13.90],
            "最低": [13.40, 13.55, 13.65],
            "成交量": [1000000, 1200000, 1100000],
            "成交额": [13500000.0, 16500000.0, 15200000.0],
            "振幅": [2.22, 2.18, 1.82],
            "涨跌幅": [0.74, 1.32, -0.44],
            "涨跌额": [0.10, 0.18, -0.06],
            "换手率": [0.85, 1.02, 0.93],
        }
    )


@pytest.fixture()
def sample_tushare_daily_df() -> pd.DataFrame:
    """模拟 Tushare daily 返回的 DataFrame"""
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["20230105", "20230104", "20230103"],
            "open": [13.80, 13.62, 13.50],
            "high": [13.90, 13.85, 13.70],
            "low": [13.65, 13.55, 13.40],
            "close": [13.72, 13.78, 13.60],
            "vol": [1100000.0, 1200000.0, 1000000.0],
            "amount": [15200.0, 16500.0, 13500.0],  # 千元
            "pre_close": [13.78, 13.60, 13.50],
            "change": [-0.06, 0.18, 0.10],
            "pct_chg": [-0.44, 1.32, 0.74],
        }
    )


@pytest.fixture()
def sample_stock_list_df() -> pd.DataFrame:
    """模拟 AKShare stock_zh_a_spot_em 返回的 DataFrame"""
    return pd.DataFrame(
        {
            "代码": ["000001", "600519", "300750"],
            "名称": ["平安银行", "贵州茅台", "宁德时代"],
            "最新价": [13.72, 1800.00, 220.50],
            "涨跌幅": [0.74, -0.55, 1.20],
        }
    )


@pytest.fixture()
def tmp_cache_dir(tmp_path: Path) -> Path:
    """临时缓存目录"""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture()
def config_with_cache(tmp_cache_dir: Path) -> AppConfig:
    """带有临时缓存目录的配置"""
    return AppConfig(
        data_source=DataSourceConfig(
            default_source="mock",
            cache_dir=str(tmp_cache_dir),
            cache_enabled=True,
        )
    )


@pytest.fixture()
def config_no_cache(tmp_cache_dir: Path) -> AppConfig:
    """禁用缓存的配置"""
    return AppConfig(
        data_source=DataSourceConfig(
            default_source="mock",
            cache_dir=str(tmp_cache_dir),
            cache_enabled=False,
        )
    )


class MockDataSource(DataSource):
    """用于测试的 Mock 数据源"""

    def __init__(self) -> None:
        self.fetch_daily_call_count = 0

    def get_name(self) -> str:
        return "mock"

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        self.fetch_daily_call_count += 1
        return pd.DataFrame(
            {
                "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "open": [13.50, 13.62, 13.80],
                "high": [13.70, 13.85, 13.90],
                "low": [13.40, 13.55, 13.65],
                "close": [13.60, 13.78, 13.72],
                "volume": [1000000, 1200000, 1100000],
                "amount": [13500000.0, 16500000.0, 15200000.0],
            }
        )

    def fetch_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": ["000001", "600519"],
                "name": ["平安银行", "贵州茅台"],
            }
        )


# ======================================================================
# Tests: DataSourceRegistry
# ======================================================================


class TestDataSourceRegistry:
    """注册表测试"""

    def test_register_and_get(self):
        """注册数据源并获取实例"""

        @DataSourceRegistry.register("test_source")
        class TestSource(DataSource):
            def get_name(self) -> str:
                return "test_source"

            def fetch_daily(self, symbol, start_date, end_date, adjust="qfq"):
                return pd.DataFrame()

            def fetch_stock_list(self):
                return pd.DataFrame()

        source = DataSourceRegistry.get("test_source")
        assert isinstance(source, TestSource)
        assert source.get_name() == "test_source"

    def test_get_unknown_source_raises(self):
        """获取未注册的数据源应抛出 ValueError"""
        with pytest.raises(ValueError, match="Unknown data source"):
            DataSourceRegistry.get("nonexistent_source")

    def test_list_sources(self):
        """列出所有已注册数据源"""

        @DataSourceRegistry.register("source_a")
        class SourceA(DataSource):
            def get_name(self):
                return "a"

            def fetch_daily(self, symbol, start_date, end_date, adjust="qfq"):
                return pd.DataFrame()

            def fetch_stock_list(self):
                return pd.DataFrame()

        @DataSourceRegistry.register("source_b")
        class SourceB(DataSource):
            def get_name(self):
                return "b"

            def fetch_daily(self, symbol, start_date, end_date, adjust="qfq"):
                return pd.DataFrame()

            def fetch_stock_list(self):
                return pd.DataFrame()

        sources = DataSourceRegistry.list_sources()
        # 应包含测试注册的和模块初始化注册的
        assert "source_a" in sources
        assert "source_b" in sources

    def test_register_returns_original_class(self):
        """装饰器应返回原始类"""

        @DataSourceRegistry.register("identity_test")
        class OriginalClass(DataSource):
            def get_name(self):
                return "identity_test"

            def fetch_daily(self, symbol, start_date, end_date, adjust="qfq"):
                return pd.DataFrame()

            def fetch_stock_list(self):
                return pd.DataFrame()

        assert DataSourceRegistry.get_class("identity_test") is OriginalClass

    def test_get_creates_new_instance(self):
        """每次 get 应返回新实例"""

        @DataSourceRegistry.register("instance_test")
        class InstanceTestSource(DataSource):
            def get_name(self):
                return "instance_test"

            def fetch_daily(self, symbol, start_date, end_date, adjust="qfq"):
                return pd.DataFrame()

            def fetch_stock_list(self):
                return pd.DataFrame()

        s1 = DataSourceRegistry.get("instance_test")
        s2 = DataSourceRegistry.get("instance_test")
        assert s1 is not s2

    def test_clear(self):
        """清除注册表"""
        DataSourceRegistry.register("temp")( type("Temp", (DataSource,), {
            "get_name": lambda self: "temp",
            "fetch_daily": lambda self, *a, **kw: pd.DataFrame(),
            "fetch_stock_list": lambda self: pd.DataFrame(),
        }))
        assert "temp" in DataSourceRegistry.list_sources()
        DataSourceRegistry.clear()
        assert DataSourceRegistry.list_sources() == []

    def test_realtime_not_implemented(self):
        """基类 fetch_realtime 默认抛出 NotImplementedError"""

        @DataSourceRegistry.register("no_realtime")
        class NoRealtimeSource(DataSource):
            def get_name(self):
                return "no_realtime"

            def fetch_daily(self, symbol, start_date, end_date, adjust="qfq"):
                return pd.DataFrame()

            def fetch_stock_list(self):
                return pd.DataFrame()

        source = DataSourceRegistry.get("no_realtime")
        with pytest.raises(NotImplementedError):
            source.fetch_realtime("000001")


# ======================================================================
# Tests: DataManager
# ======================================================================


class TestDataManager:
    """数据管理器测试"""

    def _register_mock(self):
        """注册 mock 数据源"""
        DataSourceRegistry.register("mock")(MockDataSource)

    def test_fetch_daily_returns_standardized(self, config_with_cache):
        """fetch_daily 返回标准化 DataFrame"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        df = dm.fetch_daily("000001", "20230101", "20231231")

        assert list(df.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["volume"].dtype == "int64"
        assert len(df) == 3

    def test_fetch_daily_sorted_by_date(self, config_with_cache):
        """数据应按日期升序排列"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        df = dm.fetch_daily("000001", "20230101", "20231231")
        dates = df["date"].tolist()
        assert dates == sorted(dates)

    def test_cache_hit(self, config_with_cache):
        """第二次调用应命中缓存"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        # 第一次调用：写入缓存
        df1 = dm.fetch_daily("000001", "20230101", "20231231")
        # 第二次调用：读取缓存
        df2 = dm.fetch_daily("000001", "20230101", "20231231")

        pd.testing.assert_frame_equal(df1, df2)

        # 验证数据源只被调用了一次
        source = dm.source
        assert isinstance(source, MockDataSource)
        assert source.fetch_daily_call_count == 1

    def test_cache_different_params(self, config_with_cache):
        """不同参数应产生不同的缓存"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        dm.fetch_daily("000001", "20230101", "20231231")
        dm.fetch_daily("600519", "20230101", "20231231")

        cache_files = list(Path(config_with_cache.data_source.cache_dir).glob("*.parquet"))
        assert len(cache_files) == 2

    def test_cache_disabled(self, config_no_cache):
        """禁用缓存时不应写入文件"""
        self._register_mock()
        dm = DataManager(config_no_cache)

        dm.fetch_daily("000001", "20230101", "20231231")

        cache_files = list(Path(config_no_cache.data_source.cache_dir).glob("*.parquet"))
        assert len(cache_files) == 0

    def test_clear_cache_all(self, config_with_cache):
        """清除全部缓存"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        dm.fetch_daily("000001", "20230101", "20231231")
        dm.fetch_daily("600519", "20230101", "20231231")

        count = dm.clear_cache()
        assert count == 2

        cache_files = list(Path(config_with_cache.data_source.cache_dir).glob("*.parquet"))
        assert len(cache_files) == 0

    def test_clear_cache_by_symbol(self, config_with_cache):
        """按股票代码清除缓存"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        dm.fetch_daily("000001", "20230101", "20231231")
        dm.fetch_daily("600519", "20230101", "20231231")

        count = dm.clear_cache(symbol="000001")
        assert count == 1

        cache_files = list(Path(config_with_cache.data_source.cache_dir).glob("*.parquet"))
        assert len(cache_files) == 1

    def test_cache_key_format(self, config_with_cache):
        """缓存文件名格式验证"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        dm.fetch_daily("000001", "2023-01-01", "2023-12-31", adjust="qfq")

        cache_files = list(Path(config_with_cache.data_source.cache_dir).glob("*.parquet"))
        assert len(cache_files) == 1
        # 验证文件名包含正确的组件（日期分隔符应被去除）
        filename = cache_files[0].name
        assert "mock" in filename
        assert "000001" in filename
        assert "20230101" in filename
        assert "20231231" in filename
        assert "qfq" in filename

    def test_fetch_stock_list(self, config_with_cache):
        """获取股票列表"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        df = dm.fetch_stock_list()
        assert "code" in df.columns
        assert "name" in df.columns
        assert len(df) == 2

    def test_set_source(self, config_with_cache):
        """切换数据源"""
        self._register_mock()
        DataSourceRegistry.register("mock2")(MockDataSource)

        dm = DataManager(config_with_cache)
        assert dm.source_name == "mock"

        dm.set_source("mock2")
        assert dm.source_name == "mock"  # MockDataSource.get_name() returns "mock"

    def test_cache_info(self, config_with_cache):
        """缓存统计信息"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        dm.fetch_daily("000001", "20230101", "20231231")

        info = dm.cache_info()
        assert info["enabled"] is True
        assert info["file_count"] == 1
        assert info["total_size_mb"] >= 0

    def test_corrupted_cache_recovery(self, config_with_cache):
        """损坏的缓存文件应自动恢复"""
        self._register_mock()
        dm = DataManager(config_with_cache)

        # 写入损坏的缓存文件
        cache_dir = Path(config_with_cache.data_source.cache_dir)
        bad_file = cache_dir / "mock_000001_20230101_20231231_qfq.parquet"
        bad_file.write_text("this is not a parquet file")

        # 应该能正常获取数据（自动清除损坏文件）
        df = dm.fetch_daily("000001", "20230101", "20231231")
        assert len(df) == 3


# ======================================================================
# Tests: AKShare Column Standardization
# ======================================================================


class TestAKShareColumnMapping:
    """AKShare 中文列名标准化测试"""

    def test_column_mapping(self, sample_akshare_df):
        """中文列名应正确映射为英文"""
        from quant_trading.data.akshare_source import _COLUMN_MAP

        df = sample_akshare_df.rename(columns=_COLUMN_MAP)

        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "amount" in df.columns

    def test_fetch_daily_with_mock(self, sample_akshare_df):
        """模拟 AKShare API 调用"""
        with patch("quant_trading.data.akshare_source._import_akshare") as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_hist.return_value = sample_akshare_df
            mock_import.return_value = mock_ak

            from quant_trading.data.akshare_source import AKShareDataSource

            source = AKShareDataSource()
            df = source.fetch_daily("000001", "20230101", "20230105")

            assert "date" in df.columns
            assert "open" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
            assert len(df) == 3

            # 验证调用参数
            mock_ak.stock_zh_a_hist.assert_called_once_with(
                symbol="000001",
                period="daily",
                start_date="20230101",
                end_date="20230105",
                adjust="qfq",
            )

    def test_fetch_daily_no_adjust(self, sample_akshare_df):
        """不复权模式"""
        with patch("quant_trading.data.akshare_source._import_akshare") as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_hist.return_value = sample_akshare_df
            mock_import.return_value = mock_ak

            from quant_trading.data.akshare_source import AKShareDataSource

            source = AKShareDataSource()
            source.fetch_daily("000001", "20230101", "20230105", adjust="none")

            mock_ak.stock_zh_a_hist.assert_called_once_with(
                symbol="000001",
                period="daily",
                start_date="20230101",
                end_date="20230105",
                adjust="",
            )

    def test_fetch_daily_invalid_adjust(self):
        """无效复权类型应报错"""
        with patch("quant_trading.data.akshare_source._import_akshare"):
            from quant_trading.data.akshare_source import AKShareDataSource

            source = AKShareDataSource()
            with pytest.raises(ValueError, match="Invalid adjust type"):
                source.fetch_daily("000001", "20230101", "20230105", adjust="invalid")

    def test_fetch_daily_empty_result(self):
        """空结果应返回空 DataFrame 但包含正确列"""
        with patch("quant_trading.data.akshare_source._import_akshare") as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_hist.return_value = pd.DataFrame()
            mock_import.return_value = mock_ak

            from quant_trading.data.akshare_source import AKShareDataSource

            source = AKShareDataSource()
            df = source.fetch_daily("000001", "20230101", "20230105")

            assert df.empty
            assert "date" in df.columns
            assert "close" in df.columns

    def test_fetch_stock_list_with_mock(self, sample_stock_list_df):
        """模拟获取股票列表"""
        with patch("quant_trading.data.akshare_source._import_akshare") as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_spot_em.return_value = sample_stock_list_df
            mock_import.return_value = mock_ak

            from quant_trading.data.akshare_source import AKShareDataSource

            source = AKShareDataSource()
            df = source.fetch_stock_list()

            assert "code" in df.columns
            assert "name" in df.columns
            assert len(df) == 3
            assert df.iloc[0]["code"] == "000001"
            assert df.iloc[1]["name"] == "贵州茅台"

    def test_fetch_daily_api_failure(self):
        """API 调用失败应抛出 RuntimeError"""
        with patch("quant_trading.data.akshare_source._import_akshare") as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_hist.side_effect = Exception("Network error")
            mock_import.return_value = mock_ak

            from quant_trading.data.akshare_source import AKShareDataSource

            source = AKShareDataSource()
            with pytest.raises(RuntimeError, match="Failed to fetch daily data"):
                source.fetch_daily("000001", "20230101", "20230105")


# ======================================================================
# Tests: Tushare Symbol Conversion
# ======================================================================


class TestTushareSymbolConversion:
    """Tushare 股票代码转换测试"""

    def test_shenzhen_main_board(self):
        """深圳主板: 000xxx -> .SZ"""
        from quant_trading.data.tushare_source import convert_symbol

        assert convert_symbol("000001") == "000001.SZ"
        assert convert_symbol("000651") == "000651.SZ"
        assert convert_symbol("000858") == "000858.SZ"

    def test_shanghai_main_board(self):
        """上海主板: 6xxxxx -> .SH"""
        from quant_trading.data.tushare_source import convert_symbol

        assert convert_symbol("600519") == "600519.SH"
        assert convert_symbol("601318") == "601318.SH"
        assert convert_symbol("688981") == "688981.SH"

    def test_chinext_board(self):
        """创业板: 3xxxxx -> .SZ"""
        from quant_trading.data.tushare_source import convert_symbol

        assert convert_symbol("300750") == "300750.SZ"
        assert convert_symbol("300059") == "300059.SZ"

    def test_beijing_exchange(self):
        """北交所: 8xxxxx, 4xxxxx -> .BJ"""
        from quant_trading.data.tushare_source import convert_symbol

        assert convert_symbol("830799") == "830799.BJ"
        assert convert_symbol("430047") == "430047.BJ"

    def test_already_formatted(self):
        """已有后缀的代码应直接返回"""
        from quant_trading.data.tushare_source import convert_symbol

        assert convert_symbol("000001.SZ") == "000001.SZ"
        assert convert_symbol("600519.SH") == "600519.SH"
        assert convert_symbol("600519.sh") == "600519.SH"  # 统一大写

    def test_strip_whitespace(self):
        """应去除前后空白"""
        from quant_trading.data.tushare_source import convert_symbol

        assert convert_symbol("  000001  ") == "000001.SZ"
        assert convert_symbol(" 600519 ") == "600519.SH"


class TestTushareDataSource:
    """Tushare 数据源测试"""

    def test_fetch_daily_with_mock(self, sample_tushare_daily_df):
        """模拟 Tushare daily API 调用"""
        with patch("quant_trading.data.tushare_source._import_tushare") as mock_import:
            mock_ts = MagicMock()
            mock_pro = MagicMock()
            mock_pro.daily.return_value = sample_tushare_daily_df
            mock_ts.pro_api.return_value = mock_pro
            mock_ts.set_token = MagicMock()
            mock_import.return_value = mock_ts

            from quant_trading.data.tushare_source import TushareDataSource

            source = TushareDataSource(token="test_token")
            df = source.fetch_daily("000001", "20230101", "20230105", adjust="none")

            assert "date" in df.columns
            assert "open" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
            assert len(df) == 3

    def test_fetch_daily_amount_conversion(self, sample_tushare_daily_df):
        """Tushare amount 从千元转换为元"""
        with patch("quant_trading.data.tushare_source._import_tushare") as mock_import:
            mock_ts = MagicMock()
            mock_pro = MagicMock()
            mock_pro.daily.return_value = sample_tushare_daily_df
            mock_ts.pro_api.return_value = mock_pro
            mock_ts.set_token = MagicMock()
            mock_import.return_value = mock_ts

            from quant_trading.data.tushare_source import TushareDataSource

            source = TushareDataSource(token="test_token")
            df = source.fetch_daily("000001", "20230101", "20230105", adjust="none")

            # 原始数据 amount 是千元 (13500.0)，转换后应为 13500000.0 元
            assert "amount" in df.columns
            assert df["amount"].iloc[0] == 15200.0 * 1000

    def test_fetch_daily_with_adjust(self, sample_tushare_daily_df):
        """使用前复权时应调用 pro_bar"""
        with patch("quant_trading.data.tushare_source._import_tushare") as mock_import:
            mock_ts = MagicMock()
            mock_pro = MagicMock()
            mock_ts.pro_api.return_value = mock_pro
            mock_ts.set_token = MagicMock()
            mock_ts.pro_bar.return_value = sample_tushare_daily_df
            mock_import.return_value = mock_ts

            from quant_trading.data.tushare_source import TushareDataSource

            source = TushareDataSource(token="test_token")
            source.fetch_daily("000001", "20230101", "20230105", adjust="qfq")

            # 前复权应使用 pro_bar 而非 daily
            mock_ts.pro_bar.assert_called_once()
            mock_pro.daily.assert_not_called()

    def test_no_token_raises(self):
        """没有 token 应抛出 ValueError"""
        with patch("quant_trading.data.tushare_source._import_tushare") as mock_import:
            mock_ts = MagicMock()
            mock_import.return_value = mock_ts

            from quant_trading.data.tushare_source import TushareDataSource

            source = TushareDataSource(token="")

            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="Tushare token is required"):
                    _ = source.pro

    def test_fetch_stock_list_with_mock(self):
        """模拟获取 Tushare 股票列表"""
        mock_df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600519.SH"],
                "symbol": ["000001", "600519"],
                "name": ["平安银行", "贵州茅台"],
                "area": ["深圳", "贵州"],
                "industry": ["银行", "白酒"],
                "market": ["主板", "主板"],
                "list_date": ["19910403", "20010827"],
            }
        )

        with patch("quant_trading.data.tushare_source._import_tushare") as mock_import:
            mock_ts = MagicMock()
            mock_pro = MagicMock()
            mock_pro.stock_basic.return_value = mock_df
            mock_ts.pro_api.return_value = mock_pro
            mock_ts.set_token = MagicMock()
            mock_import.return_value = mock_ts

            from quant_trading.data.tushare_source import TushareDataSource

            source = TushareDataSource(token="test_token")
            df = source.fetch_stock_list()

            assert "code" in df.columns
            assert "name" in df.columns
            assert df.iloc[0]["code"] == "000001"
            assert df.iloc[1]["name"] == "贵州茅台"


# ======================================================================
# Tests: DataManager Standardization
# ======================================================================


class TestStandardization:
    """DataManager._standardize 方法测试"""

    def test_standardize_types(self):
        """数据类型应正确转换"""
        df = pd.DataFrame(
            {
                "date": ["2023-01-03", "2023-01-04"],
                "open": ["13.50", "13.62"],
                "high": ["13.70", "13.85"],
                "low": ["13.40", "13.55"],
                "close": ["13.60", "13.78"],
                "volume": ["1000000", "1200000"],
            }
        )
        result = DataManager._standardize(df)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert pd.api.types.is_float_dtype(result["open"])
        assert result["volume"].dtype == "int64"

    def test_standardize_preserves_amount(self):
        """标准化应保留可选的 amount 列"""
        df = pd.DataFrame(
            {
                "date": ["2023-01-03"],
                "open": [13.50],
                "high": [13.70],
                "low": [13.40],
                "close": [13.60],
                "volume": [1000000],
                "amount": [13500000.0],
            }
        )
        result = DataManager._standardize(df)
        assert "amount" in result.columns

    def test_standardize_drops_extra_columns(self):
        """非标准列应被丢弃"""
        df = pd.DataFrame(
            {
                "date": ["2023-01-03"],
                "open": [13.50],
                "high": [13.70],
                "low": [13.40],
                "close": [13.60],
                "volume": [1000000],
                "extra_col": ["should_be_dropped"],
                "another_extra": [42],
            }
        )
        result = DataManager._standardize(df)
        assert "extra_col" not in result.columns
        assert "another_extra" not in result.columns

    def test_standardize_missing_required_columns(self):
        """缺少必需列应抛出 ValueError"""
        df = pd.DataFrame(
            {
                "date": ["2023-01-03"],
                "open": [13.50],
                # missing high, low, close, volume
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            DataManager._standardize(df)

    def test_standardize_empty_dataframe(self):
        """空 DataFrame 应返回空 DataFrame"""
        df = pd.DataFrame()
        result = DataManager._standardize(df)
        assert result.empty

    def test_standardize_sort_by_date(self):
        """应按日期升序排列"""
        df = pd.DataFrame(
            {
                "date": ["2023-01-05", "2023-01-03", "2023-01-04"],
                "open": [13.80, 13.50, 13.62],
                "high": [13.90, 13.70, 13.85],
                "low": [13.65, 13.40, 13.55],
                "close": [13.72, 13.60, 13.78],
                "volume": [1100000, 1000000, 1200000],
            }
        )
        result = DataManager._standardize(df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)
        # 第一行应是最早的日期
        assert result.iloc[0]["open"] == 13.50


# ======================================================================
# Tests: Cache File Operations
# ======================================================================


class TestCacheFileOperations:
    """Parquet 缓存文件读写测试"""

    def test_write_and_read_parquet(self, tmp_path):
        """Parquet 读写往返一致性"""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
                "open": [13.50, 13.62],
                "high": [13.70, 13.85],
                "low": [13.40, 13.55],
                "close": [13.60, 13.78],
                "volume": [1000000, 1200000],
            }
        )
        # volume 统一为 int64
        df["volume"] = df["volume"].astype("int64")

        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        loaded = pd.read_parquet(path)

        pd.testing.assert_frame_equal(df, loaded)

    def test_cache_key_generation(self, config_with_cache):
        """缓存键生成测试"""
        DataSourceRegistry.register("mock")(MockDataSource)
        dm = DataManager(config_with_cache)

        key = dm._cache_key("akshare", "000001", "2023-01-01", "2023-12-31", "qfq")
        expected_name = "akshare_000001_20230101_20231231_qfq.parquet"
        assert key.name == expected_name
        assert key.parent == Path(config_with_cache.data_source.cache_dir)

    def test_cache_key_already_no_dashes(self, config_with_cache):
        """日期已无分隔符时缓存键应正确"""
        DataSourceRegistry.register("mock")(MockDataSource)
        dm = DataManager(config_with_cache)

        key = dm._cache_key("akshare", "000001", "20230101", "20231231", "qfq")
        expected_name = "akshare_000001_20230101_20231231_qfq.parquet"
        assert key.name == expected_name


# ======================================================================
# Tests: Integration-style (mock API, full DataManager flow)
# ======================================================================


class TestDataManagerIntegration:
    """DataManager 集成测试（使用 mock）"""

    def test_full_flow_akshare(self, sample_akshare_df, tmp_path):
        """完整流程: AKShare fetch -> standardize -> cache -> re-read"""
        with patch("quant_trading.data.akshare_source._import_akshare") as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_zh_a_hist.return_value = sample_akshare_df
            mock_import.return_value = mock_ak

            config = AppConfig(
                data_source=DataSourceConfig(
                    default_source="akshare",
                    cache_dir=str(tmp_path / "cache"),
                    cache_enabled=True,
                )
            )
            dm = DataManager(config)

            # 第一次获取
            df1 = dm.fetch_daily("000001", "20230101", "20230105")
            assert len(df1) == 3
            assert pd.api.types.is_datetime64_any_dtype(df1["date"])

            # 第二次获取（应命中缓存）
            df2 = dm.fetch_daily("000001", "20230101", "20230105")
            pd.testing.assert_frame_equal(df1, df2)

            # AKShare 只应被调用一次
            assert mock_ak.stock_zh_a_hist.call_count == 1

            # 清除缓存后再次获取
            dm.clear_cache()
            df3 = dm.fetch_daily("000001", "20230101", "20230105")
            assert mock_ak.stock_zh_a_hist.call_count == 2
            pd.testing.assert_frame_equal(df1, df3)

    def test_full_flow_tushare(self, sample_tushare_daily_df, tmp_path):
        """完整流程: Tushare fetch -> standardize -> cache"""
        with patch("quant_trading.data.tushare_source._import_tushare") as mock_import, \
             patch.dict("os.environ", {"TUSHARE_TOKEN": "test_token_123"}):
            mock_ts = MagicMock()
            mock_pro = MagicMock()
            mock_pro.daily.return_value = sample_tushare_daily_df
            mock_ts.pro_api.return_value = mock_pro
            mock_ts.set_token = MagicMock()
            mock_import.return_value = mock_ts

            config = AppConfig(
                data_source=DataSourceConfig(
                    default_source="tushare",
                    cache_dir=str(tmp_path / "cache"),
                    cache_enabled=True,
                )
            )
            dm = DataManager(config)

            df = dm.fetch_daily("000001", "20230101", "20230105", adjust="none")
            assert len(df) == 3
            assert pd.api.types.is_datetime64_any_dtype(df["date"])

            # 验证 amount 已从千元转为元
            assert "amount" in df.columns
