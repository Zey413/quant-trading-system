"""日志配置 - 统一的日志管理

提供统一的日志格式与配置，支持控制台彩色输出和可选的文件日志。

Usage::

    from quant_trading.core.logging_config import setup_logging

    setup_logging(level="DEBUG", log_file="output/backtest.log")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ANSI 颜色码
_COLORS = {
    "DEBUG": "\033[36m",     # 青色
    "INFO": "\033[32m",      # 绿色
    "WARNING": "\033[33m",   # 黄色
    "ERROR": "\033[31m",     # 红色
    "CRITICAL": "\033[35m",  # 紫色
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器

    格式: 2024-01-01 09:30:00 [INFO] quant_trading.backtest.engine: 开始回测
    """

    def __init__(self, fmt: str | None = None, use_color: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if self.use_color and levelname in _COLORS:
            color = _COLORS[levelname]
            record.levelname = f"{color}{levelname}{_RESET}"
        result = super().format(record)
        # 恢复原始 levelname 以避免副作用
        record.levelname = levelname
        return result


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    use_color: bool = True,
) -> None:
    """配置日志

    为整个 quant_trading 包设置统一的日志格式和级别。

    - 控制台处理器（stderr）始终添加，支持 ANSI 彩色输出。
    - 可选择同时写入日志文件（纯文本，无颜色）。

    Args:
        level: 日志级别，可选 DEBUG / INFO / WARNING / ERROR / CRITICAL。
        log_file: 可选的日志文件路径。如果提供，会自动创建父目录。
        use_color: 控制台是否使用彩色输出（默认 True）。

    Example:
        >>> setup_logging(level="DEBUG", log_file="output/backtest.log")
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {level!r}")

    # 获取 quant_trading 根 logger
    root_logger = logging.getLogger("quant_trading")
    root_logger.setLevel(numeric_level)

    # 移除已有的处理器（避免重复调用 setup_logging 导致重复输出）
    root_logger.handlers.clear()

    # --- 控制台处理器 ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    # 仅在终端环境中使用颜色
    effective_color = use_color and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    console_formatter = _ColorFormatter(fmt=_LOG_FORMAT, use_color=effective_color)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # --- 可选文件处理器 ---
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            str(log_path), mode="a", encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            fmt=_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # 防止日志向上层 root logger 传播导致重复输出
    root_logger.propagate = False
