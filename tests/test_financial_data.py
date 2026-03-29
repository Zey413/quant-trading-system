"""财务数据模块单元测试

测试覆盖:
- FinancialDataProvider 的所有公开方法
- StockInfo / DividendRecord 数据模型
- 财务报表获取（利润表、资产负债表、现金流量表）
- 个股信息获取
- 行业成分股与指数成分股
- 分红历史
- 异常处理与边界条件
- 不依赖真实 API 调用（全部 mock）
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_trading.data.financial import (
    DividendRecord,
    FinancialDataProvider,
    REPORT_TYPES,
    StockInfo,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def sample_income_df() -> pd.DataFrame:
    """模拟利润表数据"""
    return pd.DataFrame(
        {
            "报告日": ["2023-12-31", "2023-09-30", "2023-06-30"],
            "营业收入": [180000000000.0, 135000000000.0, 90000000000.0],
            "营业利润": [50000000000.0, 37000000000.0, 25000000000.0],
            "利润总额": [48000000000.0, 36000000000.0, 24000000000.0],
            "净利润": [38000000000.0, 28000000000.0, 19000000000.0],
        }
    )


@pytest.fixture()
def sample_balance_df() -> pd.DataFrame:
    """模拟资产负债表数据"""
    return pd.DataFrame(
        {
            "报告日": ["2023-12-31", "2023-09-30"],
            "总资产": [5500000000000.0, 5300000000000.0],
            "总负债": [5000000000000.0, 4800000000000.0],
            "股东权益": [500000000000.0, 480000000000.0],
        }
    )


@pytest.fixture()
def sample_cashflow_df() -> pd.DataFrame:
    """模拟现金流量表数据"""
    return pd.DataFrame(
        {
            "报告日": ["2023-12-31", "2023-09-30"],
            "经营现金流入": [200000000000.0, 150000000000.0],
            "经营现金流出": [180000000000.0, 135000000000.0],
            "经营现金流净额": [20000000000.0, 15000000000.0],
        }
    )


@pytest.fixture()
def sample_stock_info_df() -> pd.DataFrame:
    """模拟 stock_individual_info_em 返回的数据"""
    return pd.DataFrame(
        {
            "item": [
                "股票代码",
                "股票简称",
                "行业",
                "上市时间",
                "总股本",
                "流通股",
                "总市值",
                "流通市值",
                "市盈率",
                "市净率",
            ],
            "value": [
                "000001",
                "平安银行",
                "银行",
                "19910403",
                1940000.0,
                1940000.0,
                266000000000.0,
                260000000000.0,
                7.50,
                0.85,
            ],
        }
    )


@pytest.fixture()
def sample_industry_stocks_df() -> pd.DataFrame:
    """模拟行业成分股数据"""
    return pd.DataFrame(
        {
            "代码": ["000001", "601398", "601288", "601939"],
            "名称": ["平安银行", "工商银行", "农业银行", "建设银行"],
            "最新价": [13.72, 5.80, 3.90, 7.20],
            "涨跌幅": [0.51, 0.17, -0.25, 0.42],
            "总市值": [266e9, 2070e9, 1360e9, 1800e9],
        }
    )


@pytest.fixture()
def sample_index_components_df() -> pd.DataFrame:
    """模拟指数成分股数据"""
    return pd.DataFrame(
        {
            "品种代码": ["600519", "000001", "601318", "300750"],
            "品种名称": ["贵州茅台", "平安银行", "中国平安", "宁德时代"],
            "纳入日期": ["2005-01-04", "2005-01-04", "2007-10-15", "2020-06-15"],
        }
    )


@pytest.fixture()
def sample_dividend_df() -> pd.DataFrame:
    """模拟分红历史数据"""
    return pd.DataFrame(
        {
            "报告期": ["2023-12-31", "2022-12-31", "2021-12-31"],
            "分红方案说明": [
                "10派7.20元",
                "10派6.80元",
                "10派6.00元",
            ],
            "除权除息日": ["2024-07-12", "2023-07-14", "2022-07-15"],
            "股权登记日": ["2024-07-11", "2023-07-13", "2022-07-14"],
            "派息(每10股)": [7.20, 6.80, 6.00],
            "送股(每10股)": [0.0, 0.0, 0.0],
            "转增(每10股)": [0.0, 0.0, 0.0],
        }
    )


@pytest.fixture()
def provider() -> FinancialDataProvider:
    """创建 FinancialDataProvider 实例"""
    return FinancialDataProvider()


# ======================================================================
# Tests: StockInfo Model
# ======================================================================


class TestStockInfoModel:
    """StockInfo 数据模型测试"""

    def test_create_stock_info(self):
        """创建基本个股信息"""
        info = StockInfo(
            symbol="000001",
            name="平安银行",
            industry="银行",
        )
        assert info.symbol == "000001"
        assert info.name == "平安银行"
        assert info.industry == "银行"

    def test_default_values(self):
        """默认值应为零或空"""
        info = StockInfo(symbol="000001")
        assert info.name == ""
        assert info.pe_ratio == 0.0
        assert info.total_market_cap == 0.0


# ======================================================================
# Tests: DividendRecord Model
# ======================================================================


class TestDividendRecordModel:
    """DividendRecord 数据模型测试"""

    def test_create_dividend_record(self):
        """创建分红记录"""
        record = DividendRecord(
            symbol="000001",
            year="2023-12-31",
            plan="10派7.20元",
            dividend_per_share=0.72,
        )
        assert record.symbol == "000001"
        assert record.dividend_per_share == 0.72

    def test_default_values(self):
        """默认值"""
        record = DividendRecord(symbol="000001")
        assert record.dividend_per_share == 0.0
        assert record.bonus_shares == 0.0


# ======================================================================
# Tests: Financial Statements
# ======================================================================


class TestFinancialStatements:
    """财务报表获取测试"""

    def test_get_income_statement(self, provider, sample_income_df):
        """获取利润表"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_financial_report_sina.return_value = sample_income_df
            mock_import.return_value = mock_ak

            df = provider.get_financial_statements("000001", "income")

            assert not df.empty
            assert len(df) == 3
            mock_ak.stock_financial_report_sina.assert_called_once_with(
                stock="000001",
                symbol="利润表",
            )

    def test_get_balance_sheet(self, provider, sample_balance_df):
        """获取资产负债表"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_financial_report_sina.return_value = sample_balance_df
            mock_import.return_value = mock_ak

            df = provider.get_financial_statements("000001", "balance")

            assert not df.empty
            mock_ak.stock_financial_report_sina.assert_called_once_with(
                stock="000001",
                symbol="资产负债表",
            )

    def test_get_cashflow_statement(self, provider, sample_cashflow_df):
        """获取现金流量表"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_financial_report_sina.return_value = sample_cashflow_df
            mock_import.return_value = mock_ak

            df = provider.get_financial_statements("000001", "cashflow")

            assert not df.empty
            mock_ak.stock_financial_report_sina.assert_called_once_with(
                stock="000001",
                symbol="现金流量表",
            )

    def test_invalid_report_type(self, provider):
        """无效报表类型应抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的 report_type"):
            provider.get_financial_statements("000001", "invalid_type")

    def test_empty_symbol(self, provider):
        """空 symbol 应抛出 ValueError"""
        with pytest.raises(ValueError, match="symbol 不能为空"):
            provider.get_financial_statements("", "income")

    def test_api_failure(self, provider):
        """API 调用失败应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_financial_report_sina.side_effect = Exception(
                "Network error"
            )
            mock_import.return_value = mock_ak

            with pytest.raises(RuntimeError, match="获取.*利润表失败"):
                provider.get_financial_statements("000001", "income")

    def test_empty_response(self, provider):
        """空响应应返回空 DataFrame"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_financial_report_sina.return_value = pd.DataFrame()
            mock_import.return_value = mock_ak

            df = provider.get_financial_statements("000001", "income")
            assert df.empty


# ======================================================================
# Tests: Stock Info
# ======================================================================


class TestGetStockInfo:
    """个股信息获取测试"""

    def test_get_stock_info_success(self, provider, sample_stock_info_df):
        """成功获取个股信息"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_individual_info_em.return_value = sample_stock_info_df
            mock_import.return_value = mock_ak

            info = provider.get_stock_info("000001")

            assert isinstance(info, StockInfo)
            assert info.symbol == "000001"
            assert info.name == "平安银行"
            assert info.industry == "银行"

    def test_stock_info_empty_symbol(self, provider):
        """空 symbol 应抛出 ValueError"""
        with pytest.raises(ValueError, match="symbol 不能为空"):
            provider.get_stock_info("")

    def test_stock_info_api_failure(self, provider):
        """API 失败应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_individual_info_em.side_effect = Exception("Timeout")
            mock_import.return_value = mock_ak

            with pytest.raises(RuntimeError, match="获取.*个股信息失败"):
                provider.get_stock_info("000001")


# ======================================================================
# Tests: Industry Stocks
# ======================================================================


class TestGetIndustryStocks:
    """行业成分股获取测试"""

    def test_get_industry_stocks_success(
        self, provider, sample_industry_stocks_df
    ):
        """成功获取行业成分股"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_board_industry_cons_em.return_value = (
                sample_industry_stocks_df
            )
            mock_import.return_value = mock_ak

            df = provider.get_industry_stocks("银行")

            assert not df.empty
            assert "code" in df.columns
            assert "name" in df.columns
            assert len(df) == 4

    def test_industry_empty_name(self, provider):
        """空行业名应抛出 ValueError"""
        with pytest.raises(ValueError, match="industry 不能为空"):
            provider.get_industry_stocks("")

    def test_industry_api_failure(self, provider):
        """API 失败应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_board_industry_cons_em.side_effect = Exception(
                "Not found"
            )
            mock_import.return_value = mock_ak

            with pytest.raises(RuntimeError, match="获取行业.*成分股失败"):
                provider.get_industry_stocks("不存在的行业")


# ======================================================================
# Tests: Index Components
# ======================================================================


class TestGetIndexComponents:
    """指数成分股获取测试"""

    def test_get_index_components_success(
        self, provider, sample_index_components_df
    ):
        """成功获取指数成分股"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.index_stock_cons.return_value = sample_index_components_df
            mock_import.return_value = mock_ak

            df = provider.get_index_components("000300")

            assert not df.empty
            assert "code" in df.columns
            assert "name" in df.columns
            assert len(df) == 4

    def test_index_empty_code(self, provider):
        """空指数代码应抛出 ValueError"""
        with pytest.raises(ValueError, match="index_code 不能为空"):
            provider.get_index_components("")

    def test_index_api_failure(self, provider):
        """API 失败应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.index_stock_cons.side_effect = Exception("Error")
            mock_import.return_value = mock_ak

            with pytest.raises(RuntimeError, match="获取指数.*成分股失败"):
                provider.get_index_components("000300")


# ======================================================================
# Tests: Dividend History
# ======================================================================


class TestGetDividendHistory:
    """分红历史获取测试"""

    def test_get_dividend_history_success(self, provider, sample_dividend_df):
        """成功获取分红历史"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_history_dividend_detail.return_value = (
                sample_dividend_df
            )
            mock_import.return_value = mock_ak

            records = provider.get_dividend_history("000001")

            assert len(records) == 3
            assert all(isinstance(r, DividendRecord) for r in records)
            assert records[0].symbol == "000001"
            # 7.20 / 10 = 0.72
            assert abs(records[0].dividend_per_share - 0.72) < 0.001

    def test_dividend_empty_symbol(self, provider):
        """空 symbol 应抛出 ValueError"""
        with pytest.raises(ValueError, match="symbol 不能为空"):
            provider.get_dividend_history("")

    def test_dividend_empty_response(self, provider):
        """空响应应返回空列表"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_history_dividend_detail.return_value = pd.DataFrame()
            mock_import.return_value = mock_ak

            records = provider.get_dividend_history("000001")
            assert records == []

    def test_dividend_api_failure(self, provider):
        """API 失败应抛出 RuntimeError"""
        with patch(
            "quant_trading.data.financial._import_akshare"
        ) as mock_import:
            mock_ak = MagicMock()
            mock_ak.stock_history_dividend_detail.side_effect = Exception(
                "Server error"
            )
            mock_import.return_value = mock_ak

            with pytest.raises(RuntimeError, match="获取.*分红历史失败"):
                provider.get_dividend_history("000001")


# ======================================================================
# Tests: Report Types & Repr
# ======================================================================


class TestReportTypesAndRepr:
    """报表类型常量与 repr 测试"""

    def test_report_types_completeness(self):
        """报表类型应包含三种"""
        assert "income" in REPORT_TYPES
        assert "balance" in REPORT_TYPES
        assert "cashflow" in REPORT_TYPES
        assert len(REPORT_TYPES) == 3

    def test_repr(self, provider):
        """repr 格式"""
        assert repr(provider) == "FinancialDataProvider()"
