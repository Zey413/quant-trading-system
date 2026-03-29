"""数据导出工具 - 支持 CSV / Excel / JSON 格式

提供统一的 DataFrame 导出接口，方便将行情数据、回测结果等导出为
常见文件格式，用于进一步分析或归档。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataExporter:
    """导出数据为各种格式

    所有方法均为 @staticmethod，可直接调用::

        DataExporter.to_csv(df, "output/data.csv")
        DataExporter.to_excel(df, "output/data.xlsx")
        DataExporter.to_json(df, "output/data.json")
    """

    @staticmethod
    def to_csv(df: pd.DataFrame, filepath: str) -> None:
        """导出 DataFrame 为 CSV

        Args:
            df: 待导出的 DataFrame
            filepath: 输出文件路径
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("数据已导出为 CSV: %s (%d 行)", filepath, len(df))

    @staticmethod
    def to_excel(df: pd.DataFrame, filepath: str) -> None:
        """导出 DataFrame 为 Excel (需要 openpyxl)

        Args:
            df: 待导出的 DataFrame
            filepath: 输出文件路径

        Raises:
            ImportError: 如果 openpyxl 未安装
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "导出 Excel 需要 openpyxl，请执行: pip install openpyxl"
            )

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False, engine="openpyxl")
        logger.info("数据已导出为 Excel: %s (%d 行)", filepath, len(df))

    @staticmethod
    def to_json(df: pd.DataFrame, filepath: str) -> None:
        """导出 DataFrame 为 JSON

        使用 ``records`` 格式，每行数据为一个 JSON 对象，
        日期类型转换为 ISO 格式字符串。

        Args:
            df: 待导出的 DataFrame
            filepath: 输出文件路径
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(
            path,
            orient="records",
            date_format="iso",
            force_ascii=False,
            indent=2,
        )
        logger.info("数据已导出为 JSON: %s (%d 行)", filepath, len(df))
