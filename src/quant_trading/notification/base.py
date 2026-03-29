"""通知基类 - 定义统一的通知接口

所有通知渠道（邮件、Webhook 等）必须继承 Notifier 基类并实现 send 方法。
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NotificationLevel(str, enum.Enum):
    """通知级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationMessage(BaseModel):
    """标准化通知消息"""
    title: str = Field(..., description="通知标题")
    content: str = Field(default="", description="通知内容（纯文本或 Markdown）")
    level: NotificationLevel = Field(
        default=NotificationLevel.INFO,
        description="通知级别",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="消息时间戳",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="附加数据（可序列化为JSON）",
    )

    def format_text(self) -> str:
        """格式化为纯文本"""
        lines = [
            f"[{self.level.value.upper()}] {self.title}",
            f"时间: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.content:
            lines.append(f"\n{self.content}")
        if self.extra:
            lines.append(f"\n附加信息: {self.extra}")
        return "\n".join(lines)

    def format_markdown(self) -> str:
        """格式化为 Markdown"""
        level_emoji = {
            NotificationLevel.DEBUG: "🔍",
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }
        emoji = level_emoji.get(self.level, "ℹ️")
        lines = [
            f"## {emoji} {self.title}",
            f"**级别:** {self.level.value.upper()}",
            f"**时间:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.content:
            lines.append(f"\n{self.content}")
        if self.extra:
            lines.append("\n**附加信息:**")
            for k, v in self.extra.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


class Notifier(ABC):
    """通知渠道抽象基类

    所有通知渠道必须实现:
    - send(): 发送通知消息
    - name (property): 渠道名称

    可选覆盖:
    - is_available(): 检查渠道是否可用
    - min_level: 最低通知级别过滤
    """

    def __init__(
        self,
        name: str = "",
        min_level: NotificationLevel = NotificationLevel.INFO,
        enabled: bool = True,
    ) -> None:
        self._name = name or self.__class__.__name__
        self.min_level = min_level
        self.enabled = enabled
        self._send_count: int = 0
        self._error_count: int = 0

    @property
    def name(self) -> str:
        """渠道名称"""
        return self._name

    @property
    def send_count(self) -> int:
        """成功发送次数"""
        return self._send_count

    @property
    def error_count(self) -> int:
        """发送失败次数"""
        return self._error_count

    def should_send(self, message: NotificationMessage) -> bool:
        """判断是否应该发送此消息

        基于启用状态和最低级别过滤。

        Parameters
        ----------
        message : NotificationMessage
            待发送消息

        Returns
        -------
        bool
            是否应该发送
        """
        if not self.enabled:
            return False

        level_order = list(NotificationLevel)
        msg_idx = level_order.index(message.level)
        min_idx = level_order.index(self.min_level)
        return msg_idx >= min_idx

    @abstractmethod
    def send(self, message: NotificationMessage) -> bool:
        """发送通知消息

        Parameters
        ----------
        message : NotificationMessage
            待发送的消息

        Returns
        -------
        bool
            是否发送成功
        """
        ...

    def safe_send(self, message: NotificationMessage) -> bool:
        """安全发送：捕获异常，记录日志

        Parameters
        ----------
        message : NotificationMessage
            待发送的消息

        Returns
        -------
        bool
            是否发送成功
        """
        if not self.should_send(message):
            logger.debug(
                "消息被过滤 [%s]: level=%s < min_level=%s",
                self.name, message.level.value, self.min_level.value,
            )
            return False

        try:
            result = self.send(message)
            if result:
                self._send_count += 1
                logger.info("通知发送成功 [%s]: %s", self.name, message.title)
            else:
                self._error_count += 1
                logger.warning("通知发送失败 [%s]: %s", self.name, message.title)
            return result
        except Exception as exc:
            self._error_count += 1
            logger.error("通知发送异常 [%s]: %s - %s", self.name, message.title, exc)
            return False

    def is_available(self) -> bool:
        """检查渠道是否可用（可被子类覆盖）"""
        return self.enabled

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"enabled={self.enabled}, min_level={self.min_level.value})"
        )
