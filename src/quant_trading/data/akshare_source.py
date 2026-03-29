"""AKShare 数据源 - 免费A股行情数据

基于 AKShare 开源库，无需注册即可获取A股日线/实时行情。
文档: https://akshare.akfamily.xyz/

注意:
- AKShare 接口返回中文列名，需映射为标准英文列名
- AKShare 有访问频率限制，高频调用可能触发限流
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_trading.data.base import DataSource, DataSourceRegistry

logger = logging.getLogger(__name__)

# AKShare 中文列名 -> 标准列名映射
_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "price_change",
    "换手率": "turnover_rate",
}

# 复权参数映射: 我们的标准 -> AKShare 参数
_ADJUST_MAP = {
    "qfq": "qfq",   # 前复权
    "hfq": "hfq",    # 后复权
    "none": "",       # 不复权
}


def _import_akshare():
    """延迟导入 akshare，并处理未安装的情况"""
    try:
        import akshare as ak
        return ak
    except ImportError:
        raise ImportError(
            "akshare is not installed. Install it with: pip install akshare"
        )


@DataSourceRegistry.register("akshare")
class AKShareDataSource(DataSource):
    """AKShare A股数据源

    使用 AKShare 的 stock_zh_a_hist 接口获取日线数据，
    使用 stock_zh_a_spot_em 接口获取股票列表/实时行情。
    """

    def get_name(self) -> str:
        return "akshare"

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
            开始日期，"YYYYMMDD" 或 "YYYY-MM-DD"
        end_date : str
            结束日期
        adjust : str
            复权类型: "qfq", "hfq", "none"

        Returns
        -------
        pd.DataFrame
            标准化列名: date, open, high, low, close, volume, amount
        """
        ak = _import_akshare()

        # 参数校验
        if adjust not in _ADJUST_MAP:
            raise ValueError(
                f"Invalid adjust type: '{adjust}'. Must be one of: {list(_ADJUST_MAP.keys())}"
            )

        # AKShare 日期格式: "YYYYMMDD"
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        # 纯数字symbol（AKShare要求）
        symbol_clean = symbol.strip()

        logger.debug(
            "AKShare fetch_daily: symbol=%s, start=%s, end=%s, adjust=%s",
            symbol_clean,
            start,
            end,
            _ADJUST_MAP[adjust],
        )

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol_clean,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=_ADJUST_MAP[adjust],
            )
        except Exception as e:
            logger.error("AKShare fetch_daily failed for %s: %s", symbol_clean, e)
            raise RuntimeError(
                f"Failed to fetch daily data for {symbol_clean} from AKShare: {e}"
            ) from e

        if df is None or df.empty:
            logger.warning("AKShare returned empty data for %s", symbol_clean)
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

        # 列名映射
        df = df.rename(columns=_COLUMN_MAP)

        # 只保留需要的列
        keep_cols = ["date", "open", "high", "low", "close", "volume", "amount"]
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols]

        return df

    def fetch_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表

        使用 stock_zh_a_spot_em (东方财富实时行情接口)
        获取全市场A股列表。

        Returns
        -------
        pd.DataFrame
            包含 code (股票代码) 和 name (股票名称) 列
        """
        ak = _import_akshare()

        try:
            df = ak.stock_zh_a_spot_em()
        except Exception as e:
            logger.error("AKShare fetch_stock_list failed: %s", e)
            raise RuntimeError(
                f"Failed to fetch stock list from AKShare: {e}"
            ) from e

        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "name"])

        # 东方财富接口列名映射
        result = pd.DataFrame()
        if "代码" in df.columns:
            result["code"] = df["代码"]
        elif "序号" in df.columns and "代码" not in df.columns:
            # 回退: 不同版本可能列名不同
            result["code"] = df.iloc[:, 1] if df.shape[1] > 1 else ""

        if "名称" in df.columns:
            result["name"] = df["名称"]
        elif df.shape[1] > 2:
            result["name"] = df.iloc[:, 2]

        # 附带额外有用的列
        col_map = {
            "最新价": "price",
            "涨跌幅": "pct_change",
            "成交量": "volume",
            "成交额": "amount",
            "总市值": "total_market_cap",
            "流通市值": "float_market_cap",
        }
        for cn_col, en_col in col_map.items():
            if cn_col in df.columns:
                result[en_col] = df[cn_col]

        return result

    def fetch_realtime(self, symbol: str) -> dict:
        """获取实时行情

        Parameters
        ----------
        symbol : str
            纯数字股票代码

        Returns
        -------
        dict
            包含 symbol, name, price, open, high, low, volume, amount 等字段
        """
        ak = _import_akshare()

        try:
            df = ak.stock_zh_a_spot_em()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch realtime data from AKShare: {e}") from e

        if df is None or df.empty:
            raise RuntimeError(f"No realtime data available for {symbol}")

        # 筛选目标股票
        code_col = "代码" if "代码" in df.columns else df.columns[1]
        row = df[df[code_col] == symbol]

        if row.empty:
            raise ValueError(f"Symbol not found: {symbol}")

        row = row.iloc[0]

        return {
            "symbol": symbol,
            "name": row.get("名称", ""),
            "price": float(row.get("最新价", 0)),
            "open": float(row.get("今开", 0)),
            "high": float(row.get("最高", 0)),
            "low": float(row.get("最低", 0)),
            "close": float(row.get("昨收", 0)),
            "volume": int(row.get("成交量", 0)),
            "amount": float(row.get("成交额", 0)),
            "pct_change": float(row.get("涨跌幅", 0)),
        }
