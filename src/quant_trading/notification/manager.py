"""通知管理器 - 统一管理多渠道通知

NotificationManager 提供:
- 多渠道注册与管理
- 统一发送接口（一条消息发送到所有注册渠道）
- 异步发送支持
- 发送历史记录
- 便捷方法（notify / alert / error）

用法::

    from quant_trading.notification import NotificationManager, EmailNotifier, DingTalkNotifier

    manager = NotificationManager()
    manager.add_notifier(EmailNotifier(...))
    manager.add_notifier(DingTalkNotifier(webhook_url="..."))

    # 发送到所有渠道
    manager.notify("买入信号", "000001 平安银行触发买入信号")

    # 仅发送到特定渠道
    manager.notify("调试信息", "...", channels=["EmailNotifier"])
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import Any

from quant_trading.notification.base import (
    Notifier,
    NotificationLevel,
    NotificationMessage,
)

logger = logging.getLogger(__name__)


class NotificationManager:
    """统一通知管理器

    Parameters
    ----------
    max_history : int
        保留的发送历史记录数量
    async_workers : int
        异步发送的线程池大小
    """

    def __init__(
        self,
        max_history: int = 100,
        async_workers: int = 4,
    ) -> None:
        self._notifiers: dict[str, Notifier] = {}
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history
        self._executor = ThreadPoolExecutor(
            max_workers=async_workers,
            thread_name_prefix="notif-worker",
        )
        logger.debug(
            "NotificationManager 初始化: max_history=%d, workers=%d",
            max_history, async_workers,
        )

    # ------------------------------------------------------------------
    # 渠道管理
    # ------------------------------------------------------------------

    def add_notifier(self, notifier: Notifier) -> None:
        """注册通知渠道

        Parameters
        ----------
        notifier : Notifier
            通知渠道实例

        Raises
        ------
        ValueError
            同名渠道已存在
        """
        with self._lock:
            if notifier.name in self._notifiers:
                raise ValueError(f"通知渠道 {notifier.name!r} 已存在")
            self._notifiers[notifier.name] = notifier
        logger.info("注册通知渠道: %s", notifier.name)

    def remove_notifier(self, name: str) -> bool:
        """移除通知渠道

        Parameters
        ----------
        name : str
            渠道名称

        Returns
        -------
        bool
            是否成功移除
        """
        with self._lock:
            if name in self._notifiers:
                del self._notifiers[name]
                logger.info("移除通知渠道: %s", name)
                return True
        return False

    def get_notifier(self, name: str) -> Notifier | None:
        """获取指定渠道

        Parameters
        ----------
        name : str
            渠道名称

        Returns
        -------
        Notifier | None
            渠道实例，不存在则返回 None
        """
        with self._lock:
            return self._notifiers.get(name)

    @property
    def notifiers(self) -> list[str]:
        """已注册的渠道名称列表"""
        with self._lock:
            return list(self._notifiers.keys())

    # ------------------------------------------------------------------
    # 发送
    # ------------------------------------------------------------------

    def send(
        self,
        message: NotificationMessage,
        channels: list[str] | None = None,
    ) -> dict[str, bool]:
        """同步发送消息到指定渠道

        Parameters
        ----------
        message : NotificationMessage
            通知消息
        channels : list[str] | None
            指定渠道名称列表。None 表示发送到所有渠道。

        Returns
        -------
        dict[str, bool]
            各渠道发送结果
        """
        with self._lock:
            if channels:
                targets = {
                    name: n for name, n in self._notifiers.items()
                    if name in channels
                }
            else:
                targets = dict(self._notifiers)

        results: dict[str, bool] = {}
        for name, notifier in targets.items():
            success = notifier.safe_send(message)
            results[name] = success

        # 记录历史
        self._add_history(message, results)

        return results

    def send_async(
        self,
        message: NotificationMessage,
        channels: list[str] | None = None,
    ) -> Future:
        """异步发送消息（在线程池中执行）

        Parameters
        ----------
        message : NotificationMessage
            通知消息
        channels : list[str] | None
            指定渠道名称列表

        Returns
        -------
        Future
            可通过 .result() 获取发送结果
        """
        return self._executor.submit(self.send, message, channels)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def notify(
        self,
        title: str,
        content: str = "",
        level: NotificationLevel = NotificationLevel.INFO,
        channels: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """快速发送通知

        Parameters
        ----------
        title : str
            标题
        content : str
            内容
        level : NotificationLevel
            级别
        channels : list[str] | None
            指定渠道
        extra : dict | None
            附加数据

        Returns
        -------
        dict[str, bool]
            各渠道发送结果
        """
        message = NotificationMessage(
            title=title,
            content=content,
            level=level,
            extra=extra or {},
        )
        return self.send(message, channels=channels)

    def alert(
        self,
        title: str,
        content: str = "",
        channels: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """发送告警通知（WARNING 级别）"""
        return self.notify(
            title, content,
            level=NotificationLevel.WARNING,
            channels=channels,
            extra=extra,
        )

    def error(
        self,
        title: str,
        content: str = "",
        channels: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """发送错误通知（ERROR 级别）"""
        return self.notify(
            title, content,
            level=NotificationLevel.ERROR,
            channels=channels,
            extra=extra,
        )

    def critical(
        self,
        title: str,
        content: str = "",
        channels: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """发送严重错误通知（CRITICAL 级别）"""
        return self.notify(
            title, content,
            level=NotificationLevel.CRITICAL,
            channels=channels,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # 历史记录
    # ------------------------------------------------------------------

    def _add_history(
        self,
        message: NotificationMessage,
        results: dict[str, bool],
    ) -> None:
        """添加发送历史"""
        record = {
            "title": message.title,
            "level": message.level.value,
            "timestamp": message.timestamp.isoformat(),
            "results": results,
            "sent_at": datetime.now().isoformat(),
        }
        self._history.append(record)
        # 保持历史记录不超过上限
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def history(self) -> list[dict[str, Any]]:
        """发送历史记录"""
        return list(self._history)

    def clear_history(self) -> None:
        """清空历史记录"""
        self._history.clear()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """获取所有渠道状态

        Returns
        -------
        dict[str, Any]
            包含各渠道状态信息
        """
        with self._lock:
            status = {}
            for name, notifier in self._notifiers.items():
                status[name] = {
                    "enabled": notifier.enabled,
                    "available": notifier.is_available(),
                    "min_level": notifier.min_level.value,
                    "send_count": notifier.send_count,
                    "error_count": notifier.error_count,
                }
            return status

    def shutdown(self) -> None:
        """关闭管理器，释放线程池资源"""
        self._executor.shutdown(wait=False)
        logger.info("NotificationManager 已关闭")

    def __repr__(self) -> str:
        return (
            f"NotificationManager(notifiers={list(self._notifiers.keys())}, "
            f"history={len(self._history)})"
        )
