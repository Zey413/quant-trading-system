"""Tushare Pro 数据源 - 专业级A股行情数据

基于 Tushare Pro 接口，需要注册并获取 API token。
文档: https://tushare.pro/document/2

注意:
- Tushare Pro 需要积分才能访问高级接口
- 日线行情(daily)为基础接口，需要较低积分
- 复权数据需使用 pro_bar 接口
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_trading.data.base import DataSource, DataSourceRegistry

logger = logging.getLogger(__name__)

# Tushare 列名 -> 标准列名映射 (日线数据)
_DAILY_COLUMN_MAP = {
    "trade_date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "vol": "volume",
    "amount": "amount",
}

# 复权参数映射
_ADJUST_MAP = {
    "qfq": "qfq",   # 前复权
    "hfq": "hfq",    # 后复权
    "none": None,     # 不复权
}


def _import_tushare():
    """延迟导入 tushare"""
    try:
        import tushare as ts
        return ts
    except ImportError:
        raise ImportError(
            "tushare is not installed. Install it with: pip install tushare"
        )


def convert_symbol(symbol: str) -> str:
    """将纯数字股票代码转换为 Tushare 格式

    规则:
    - 以 6 开头 -> 上海交易所，添加 .SH 后缀
    - 以 0, 3 开头 -> 深圳交易所，添加 .SZ 后缀
    - 以 8, 4 开头 -> 北交所，添加 .BJ 后缀

    Parameters
    ----------
    symbol : str
        纯数字股票代码，例如 "000001", "600519", "300750"

    Returns
    -------
    str
        Tushare 格式，例如 "000001.SZ", "600519.SH"

    Examples
    --------
    >>> convert_symbol("000001")
    '000001.SZ'
    >>> convert_symbol("600519")
    '600519.SH'
    >>> convert_symbol("300750")
    '300750.SZ'
    >>> convert_symbol("830799")
    '830799.BJ'
    """
    symbol = symbol.strip()
    if "." in symbol:
        # 已经包含后缀，直接返回
        return symbol.upper()

    if symbol.startswith("6"):
        return f"{symbol}.SH"
    elif symbol.startswith(("0", "3")):
        return f"{symbol}.SZ"
    elif symbol.startswith(("8", "4")):
        return f"{symbol}.BJ"
    else:
        # 默认深圳
        logger.warning("Unknown symbol prefix for '%s', defaulting to .SZ", symbol)
        return f"{symbol}.SZ"


@DataSourceRegistry.register("tushare")
class TushareDataSource(DataSource):
    """Tushare Pro A股数据源

    Parameters
    ----------
    token : str, optional
        Tushare Pro API token。如不提供，会尝试从环境变量
        TUSHARE_TOKEN 或配置文件中读取。
    """

    def __init__(self, token: str = "") -> None:
        self._token = token
        self._pro = None

    @property
    def pro(self):
        """延迟初始化 Tushare Pro API 接口"""
        if self._pro is None:
            ts = _import_tushare()

            token = self._token
            if not token:
                # 尝试从环境变量获取
                import os
                token = os.environ.get("TUSHARE_TOKEN", "")

            if not token:
                raise ValueError(
                    "Tushare token is required. Set it via:\n"
                    "  1. TushareDataSource(token='your_token')\n"
                    "  2. Environment variable TUSHARE_TOKEN\n"
                    "  3. Config file: data_source.tushare_token"
                )

            ts.set_token(token)
            self._pro = ts.pro_api()
            logger.info("Tushare Pro API initialized")

        return self._pro

    def get_name(self) -> str:
        return "tushare"

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线数据

        Parameters
        ----------
        symbol : str
            纯数字股票代码，例如 "000001", "600519"
        start_date : str
            开始日期
        end_date : str
            结束日期
        adjust : str
            复权类型: "qfq", "hfq", "none"

        Returns
        -------
        pd.DataFrame
            标准化列名: date, open, high, low, close, volume, amount
        """
        ts = _import_tushare()

        if adjust not in _ADJUST_MAP:
            raise ValueError(
                f"Invalid adjust type: '{adjust}'. Must be one of: {list(_ADJUST_MAP.keys())}"
            )

        # 转换代码格式
        ts_code = convert_symbol(symbol)

        # 日期格式: Tushare 使用 "YYYYMMDD"
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        logger.debug(
            "Tushare fetch_daily: ts_code=%s, start=%s, end=%s, adjust=%s",
            ts_code,
            start,
            end,
            adjust,
        )

        adj = _ADJUST_MAP[adjust]

        try:
            if adj is not None:
                # 使用 pro_bar 获取复权数据
                df = ts.pro_bar(
                    ts_code=ts_code,
                    adj=adj,
                    asset="E",
                    freq="D",
                    start_date=start,
                    end_date=end,
                )
            else:
                # 不复权，使用 daily 接口
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                )
        except Exception as e:
            logger.error("Tushare fetch_daily failed for %s: %s", ts_code, e)
            raise RuntimeError(
                f"Failed to fetch daily data for {ts_code} from Tushare: {e}"
            ) from e

        if df is None or df.empty:
            logger.warning("Tushare returned empty data for %s", ts_code)
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount"]
            )

        # 列名映射
        df = df.rename(columns=_DAILY_COLUMN_MAP)

        # 只保留需要的列
        keep_cols = ["date", "open", "high", "low", "close", "volume", "amount"]
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols]

        # Tushare 的 amount 单位是千元，转换为元
        if "amount" in df.columns:
            df["amount"] = df["amount"] * 1000

        return df

    def fetch_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表

        使用 stock_basic 接口获取上市公司列表。

        Returns
        -------
        pd.DataFrame
            包含 code, name 以及其他基本信息
        """
        try:
            df = self.pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date",
            )
        except Exception as e:
            logger.error("Tushare fetch_stock_list failed: %s", e)
            raise RuntimeError(
                f"Failed to fetch stock list from Tushare: {e}"
            ) from e

        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "name"])

        result = pd.DataFrame()
        result["code"] = df["symbol"] if "symbol" in df.columns else df["ts_code"].str[:6]
        result["name"] = df.get("name", "")

        # 附带额外信息
        optional_fields = {
            "ts_code": "ts_code",
            "area": "area",
            "industry": "industry",
            "market": "market",
            "list_date": "list_date",
        }
        for src_col, dst_col in optional_fields.items():
            if src_col in df.columns:
                result[dst_col] = df[src_col]

        return result

    def fetch_realtime(self, symbol: str) -> dict:
        """获取实时行情（Tushare 基础版不直接支持）

        Tushare Pro 的实时行情需要较高积分，这里使用
        当日日线数据作为替代。
        """
        ts_code = convert_symbol(symbol)
        from datetime import date

        today = date.today().strftime("%Y%m%d")

        try:
            df = self.pro.daily(ts_code=ts_code, start_date=today, end_date=today)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch realtime data for {ts_code}: {e}") from e

        if df is None or df.empty:
            raise RuntimeError(f"No realtime data available for {symbol}")

        row = df.iloc[0]
        return {
            "symbol": symbol,
            "name": "",
            "price": float(row.get("close", 0)),
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("pre_close", 0)),
            "volume": int(row.get("vol", 0)),
            "amount": float(row.get("amount", 0)) * 1000,
            "pct_change": float(row.get("pct_chg", 0)),
        }
