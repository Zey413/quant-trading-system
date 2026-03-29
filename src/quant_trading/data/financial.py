"""财务数据模块 - 基于AKShare的A股财务数据获取

提供:
- FinancialDataProvider: 财务数据提供者，支持财务报表、公司信息、行业数据等

支持的数据类型:
- 财务报表（利润表、资产负债表、现金流量表）
- 个股信息（公司概况）
- 行业分类与成分股
- 指数成分股
- 分红历史

用法::

    from quant_trading.data.financial import FinancialDataProvider

    provider = FinancialDataProvider()

    # 获取财务报表
    df = provider.get_financial_statements("000001", report_type="income")

    # 获取个股信息
    info = provider.get_stock_info("000001")

    # 获取行业成分股
    stocks = provider.get_industry_stocks("银行")

    # 获取指数成分股
    components = provider.get_index_components("000300")

    # 获取分红历史
    dividends = provider.get_dividend_history("000001")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ======================================================================
# AKShare 延迟导入
# ======================================================================


def _import_akshare():
    """延迟导入 akshare，并处理未安装的情况"""
    try:
        import akshare as ak
        return ak
    except ImportError:
        raise ImportError(
            "akshare is not installed. Install it with: pip install akshare"
        )


# ======================================================================
# 数据模型
# ======================================================================


class StockInfo(BaseModel):
    """个股基本信息

    Attributes
    ----------
    symbol : str
        股票代码
    name : str
        股票名称
    industry : str
        所属行业
    area : str
        所在地区
    market : str
        上市板块（主板/创业板/科创板）
    list_date : str
        上市日期
    total_shares : float
        总股本（万股）
    float_shares : float
        流通股本（万股）
    total_market_cap : float
        总市值（元）
    float_market_cap : float
        流通市值（元）
    pe_ratio : float
        市盈率
    pb_ratio : float
        市净率
    """

    symbol: str = Field(..., description="股票代码")
    name: str = Field(default="", description="股票名称")
    industry: str = Field(default="", description="所属行业")
    area: str = Field(default="", description="所在地区")
    market: str = Field(default="", description="上市板块")
    list_date: str = Field(default="", description="上市日期")
    total_shares: float = Field(default=0.0, description="总股本（万股）")
    float_shares: float = Field(default=0.0, description="流通股本（万股）")
    total_market_cap: float = Field(default=0.0, description="总市值（元）")
    float_market_cap: float = Field(default=0.0, description="流通市值（元）")
    pe_ratio: float = Field(default=0.0, description="市盈率")
    pb_ratio: float = Field(default=0.0, description="市净率")


class DividendRecord(BaseModel):
    """分红记录

    Attributes
    ----------
    symbol : str
        股票代码
    year : str
        分红年度
    plan : str
        分红方案描述
    ex_date : str
        除权除息日
    record_date : str
        股权登记日
    dividend_per_share : float
        每股分红（元）
    bonus_shares : float
        每股送股（股）
    convert_shares : float
        每股转增（股）
    """

    symbol: str = Field(..., description="股票代码")
    year: str = Field(default="", description="分红年度")
    plan: str = Field(default="", description="分红方案")
    ex_date: str = Field(default="", description="除权除息日")
    record_date: str = Field(default="", description="股权登记日")
    dividend_per_share: float = Field(default=0.0, description="每股分红（元）")
    bonus_shares: float = Field(default=0.0, description="每股送股")
    convert_shares: float = Field(default=0.0, description="每股转增")


# ======================================================================
# 报表类型枚举
# ======================================================================

# 支持的财务报表类型
REPORT_TYPES = {
    "income": "利润表",
    "balance": "资产负债表",
    "cashflow": "现金流量表",
}


# ======================================================================
# 财务数据提供者
# ======================================================================


class FinancialDataProvider:
    """财务数据提供者 - 基于AKShare

    集成 AKShare 的财务数据接口，提供统一的财务数据获取API。
    支持财务报表、个股信息、行业数据、指数成分股和分红历史等。

    Notes
    -----
    - AKShare 接口返回中文列名，本模块保留原始列名不做映射
    - AKShare 有访问频率限制，高频调用可能触发限流
    - 建议配合 rate_limiter 和 cache 模块使用

    Examples
    --------
    >>> provider = FinancialDataProvider()
    >>> df = provider.get_financial_statements("000001", "income")
    >>> info = provider.get_stock_info("000001")
    >>> print(info.name, info.industry)
    平安银行 银行
    """

    def __init__(self) -> None:
        logger.debug("FinancialDataProvider 初始化")

    def get_financial_statements(
        self,
        symbol: str,
        report_type: str = "income",
    ) -> pd.DataFrame:
        """获取财务报表数据

        Parameters
        ----------
        symbol : str
            股票代码（纯数字，如 "000001"）
        report_type : str
            报表类型:
            - "income": 利润表
            - "balance": 资产负债表
            - "cashflow": 现金流量表

        Returns
        -------
        pd.DataFrame
            财务报表数据，保留 AKShare 原始列名

        Raises
        ------
        ValueError
            无效的 report_type 或 symbol
        RuntimeError
            API 调用失败
        """
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol 不能为空")

        if report_type not in REPORT_TYPES:
            raise ValueError(
                f"无效的 report_type: '{report_type}'。"
                f"支持的类型: {list(REPORT_TYPES.keys())}"
            )

        ak = _import_akshare()
        report_name = REPORT_TYPES[report_type]

        logger.debug(
            "获取财务报表: symbol=%s, type=%s (%s)",
            symbol,
            report_type,
            report_name,
        )

        try:
            if report_type == "income":
                df = ak.stock_financial_report_sina(
                    stock=symbol,
                    symbol="利润表",
                )
            elif report_type == "balance":
                df = ak.stock_financial_report_sina(
                    stock=symbol,
                    symbol="资产负债表",
                )
            elif report_type == "cashflow":
                df = ak.stock_financial_report_sina(
                    stock=symbol,
                    symbol="现金流量表",
                )
            else:
                # 不应到达这里，但作为防御性编程
                raise ValueError(f"未知的报表类型: {report_type}")

        except ImportError:
            raise
        except ValueError:
            raise
        except Exception as exc:
            logger.error(
                "获取财务报表失败 [%s, %s]: %s",
                symbol,
                report_type,
                exc,
            )
            raise RuntimeError(
                f"获取 {symbol} 的{report_name}失败: {exc}"
            ) from exc

        if df is None or df.empty:
            logger.warning(
                "财务报表数据为空: symbol=%s, type=%s",
                symbol,
                report_type,
            )
            return pd.DataFrame()

        logger.debug(
            "获取财务报表成功: symbol=%s, type=%s, rows=%d",
            symbol,
            report_type,
            len(df),
        )
        return df

    def get_stock_info(self, symbol: str) -> StockInfo:
        """获取个股基本信息

        基于 AKShare 的 stock_individual_info_em 接口获取公司概况。

        Parameters
        ----------
        symbol : str
            股票代码（纯数字，如 "000001"）

        Returns
        -------
        StockInfo
            个股基本信息

        Raises
        ------
        ValueError
            无效的 symbol
        RuntimeError
            API 调用失败
        """
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol 不能为空")

        ak = _import_akshare()
        logger.debug("获取个股信息: %s", symbol)

        try:
            df = ak.stock_individual_info_em(symbol=symbol)
        except ImportError:
            raise
        except Exception as exc:
            logger.error("获取个股信息失败 [%s]: %s", symbol, exc)
            raise RuntimeError(
                f"获取 {symbol} 的个股信息失败: {exc}"
            ) from exc

        if df is None or df.empty:
            raise RuntimeError(f"个股信息为空: {symbol}")

        # stock_individual_info_em 返回两列: item(字段名), value(字段值)
        info_dict: dict[str, Any] = {}
        if "item" in df.columns and "value" in df.columns:
            for _, row in df.iterrows():
                info_dict[str(row["item"])] = row["value"]
        else:
            # 兼容不同版本的AKShare
            for _, row in df.iterrows():
                info_dict[str(row.iloc[0])] = row.iloc[1]

        # 映射到 StockInfo 字段
        field_map: dict[str, str] = {
            "股票代码": "symbol",
            "股票简称": "name",
            "行业": "industry",
            "地区": "area",  # 部分版本可能叫"省份"
            "上市时间": "list_date",
            "总股本": "total_shares",
            "流通股": "float_shares",
            "总市值": "total_market_cap",
            "流通市值": "float_market_cap",
            "市盈率": "pe_ratio",
            "市净率": "pb_ratio",
        }

        stock_data: dict[str, Any] = {"symbol": symbol}
        for cn_key, en_key in field_map.items():
            if cn_key in info_dict:
                val = info_dict[cn_key]
                if val is not None and val != "-":
                    stock_data[en_key] = val

        stock_info = StockInfo(**stock_data)
        logger.debug("获取个股信息成功: %s (%s)", symbol, stock_info.name)
        return stock_info

    def get_industry_stocks(self, industry: str) -> pd.DataFrame:
        """获取指定行业的成分股列表

        Parameters
        ----------
        industry : str
            行业名称（如 "银行"、"白酒"）

        Returns
        -------
        pd.DataFrame
            行业成分股列表，包含股票代码和名称

        Raises
        ------
        ValueError
            无效的 industry
        RuntimeError
            API 调用失败
        """
        industry = industry.strip()
        if not industry:
            raise ValueError("industry 不能为空")

        ak = _import_akshare()
        logger.debug("获取行业成分股: %s", industry)

        try:
            df = ak.stock_board_industry_cons_em(symbol=industry)
        except ImportError:
            raise
        except Exception as exc:
            logger.error("获取行业成分股失败 [%s]: %s", industry, exc)
            raise RuntimeError(
                f"获取行业 '{industry}' 的成分股失败: {exc}"
            ) from exc

        if df is None or df.empty:
            logger.warning("行业成分股数据为空: %s", industry)
            return pd.DataFrame(columns=["code", "name"])

        # 标准化列名
        result = pd.DataFrame()
        if "代码" in df.columns:
            result["code"] = df["代码"]
        if "名称" in df.columns:
            result["name"] = df["名称"]

        # 附加额外列
        extra_map = {
            "最新价": "price",
            "涨跌幅": "pct_change",
            "成交量": "volume",
            "成交额": "amount",
            "总市值": "total_market_cap",
            "流通市值": "float_market_cap",
        }
        for cn_col, en_col in extra_map.items():
            if cn_col in df.columns:
                result[en_col] = df[cn_col]

        logger.debug(
            "获取行业成分股成功: %s, %d 只",
            industry,
            len(result),
        )
        return result

    def get_index_components(self, index_code: str) -> pd.DataFrame:
        """获取指数成分股

        Parameters
        ----------
        index_code : str
            指数代码，如:
            - "000300": 沪深300
            - "000905": 中证500
            - "000016": 上证50
            - "399006": 创业板指

        Returns
        -------
        pd.DataFrame
            指数成分股列表

        Raises
        ------
        ValueError
            无效的 index_code
        RuntimeError
            API 调用失败
        """
        index_code = index_code.strip()
        if not index_code:
            raise ValueError("index_code 不能为空")

        ak = _import_akshare()
        logger.debug("获取指数成分股: %s", index_code)

        try:
            df = ak.index_stock_cons(symbol=index_code)
        except ImportError:
            raise
        except Exception as exc:
            logger.error("获取指数成分股失败 [%s]: %s", index_code, exc)
            raise RuntimeError(
                f"获取指数 '{index_code}' 的成分股失败: {exc}"
            ) from exc

        if df is None or df.empty:
            logger.warning("指数成分股数据为空: %s", index_code)
            return pd.DataFrame(columns=["code", "name"])

        # 标准化列名
        result = pd.DataFrame()
        col_map = {
            "品种代码": "code",
            "品种名称": "name",
            "纳入日期": "include_date",
        }
        for cn_col, en_col in col_map.items():
            if cn_col in df.columns:
                result[en_col] = df[cn_col]

        # 兼容不同版本
        if "code" not in result.columns and "代码" in df.columns:
            result["code"] = df["代码"]
        if "name" not in result.columns and "名称" in df.columns:
            result["name"] = df["名称"]

        logger.debug(
            "获取指数成分股成功: %s, %d 只",
            index_code,
            len(result),
        )
        return result

    def get_dividend_history(self, symbol: str) -> list[DividendRecord]:
        """获取分红历史

        Parameters
        ----------
        symbol : str
            股票代码（纯数字，如 "000001"）

        Returns
        -------
        list[DividendRecord]
            分红记录列表，按时间倒序排列

        Raises
        ------
        ValueError
            无效的 symbol
        RuntimeError
            API 调用失败
        """
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol 不能为空")

        ak = _import_akshare()
        logger.debug("获取分红历史: %s", symbol)

        try:
            df = ak.stock_history_dividend_detail(
                symbol=symbol,
                indicator="分红",
            )
        except ImportError:
            raise
        except Exception as exc:
            logger.error("获取分红历史失败 [%s]: %s", symbol, exc)
            raise RuntimeError(
                f"获取 {symbol} 的分红历史失败: {exc}"
            ) from exc

        if df is None or df.empty:
            logger.warning("分红历史数据为空: %s", symbol)
            return []

        # 解析分红记录
        records: list[DividendRecord] = []
        for _, row in df.iterrows():
            try:
                record = DividendRecord(
                    symbol=symbol,
                    year=str(row.get("报告期", "")),
                    plan=str(row.get("分红方案说明", row.get("分配方案", ""))),
                    ex_date=str(row.get("除权除息日", "")),
                    record_date=str(row.get("股权登记日", "")),
                    dividend_per_share=float(row.get("派息(每10股)", 0)) / 10.0,
                    bonus_shares=float(row.get("送股(每10股)", 0)) / 10.0,
                    convert_shares=float(row.get("转增(每10股)", 0)) / 10.0,
                )
                records.append(record)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "解析分红记录失败 [%s]: %s (row=%s)",
                    symbol,
                    exc,
                    dict(row),
                )
                continue

        logger.debug("获取分红历史成功: %s, %d 条记录", symbol, len(records))
        return records

    def get_industry_list(self) -> pd.DataFrame:
        """获取行业板块列表

        Returns
        -------
        pd.DataFrame
            行业板块列表

        Raises
        ------
        RuntimeError
            API 调用失败
        """
        ak = _import_akshare()
        logger.debug("获取行业板块列表")

        try:
            df = ak.stock_board_industry_name_em()
        except ImportError:
            raise
        except Exception as exc:
            logger.error("获取行业板块列表失败: %s", exc)
            raise RuntimeError(
                f"获取行业板块列表失败: {exc}"
            ) from exc

        if df is None or df.empty:
            logger.warning("行业板块列表为空")
            return pd.DataFrame(columns=["name", "code"])

        # 标准化列名
        result = pd.DataFrame()
        if "板块名称" in df.columns:
            result["name"] = df["板块名称"]
        if "板块代码" in df.columns:
            result["code"] = df["板块代码"]

        # 附加额外列
        extra_map = {
            "最新价": "price",
            "涨跌幅": "pct_change",
            "成交量": "volume",
            "成交额": "amount",
        }
        for cn_col, en_col in extra_map.items():
            if cn_col in df.columns:
                result[en_col] = df[cn_col]

        logger.debug("获取行业板块列表成功: %d 个行业", len(result))
        return result

    def __repr__(self) -> str:
        return "FinancialDataProvider()"
