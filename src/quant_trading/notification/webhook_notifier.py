"""Webhook 通知 - 钉钉/企业微信/Slack 推送

支持多种 Webhook 平台，每个平台使用对应的消息格式。

用法::

    from quant_trading.notification.webhook_notifier import (
        DingTalkNotifier, WeChatWorkNotifier, SlackNotifier,
    )

    # 钉钉
    ding = DingTalkNotifier(webhook_url="https://oapi.dingtalk.com/robot/send?access_token=xxx")
    # 企业微信
    wechat = WeChatWorkNotifier(webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx")
    # Slack
    slack = SlackNotifier(webhook_url="https://hooks.slack.com/services/xxx/yyy/zzz")
"""

from __future__ import annotations

import hashlib
import hmac
import base64
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any

from quant_trading.notification.base import (
    Notifier,
    NotificationLevel,
    NotificationMessage,
)

logger = logging.getLogger(__name__)


class WebhookNotifier(Notifier):
    """通用 Webhook 通知基类

    通过 HTTP POST 向指定 URL 发送 JSON 消息。

    Parameters
    ----------
    webhook_url : str
        Webhook 地址
    timeout : int
        HTTP 请求超时时间（秒）
    min_level : NotificationLevel
        最低通知级别
    enabled : bool
        是否启用
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = 10,
        name: str = "",
        min_level: NotificationLevel = NotificationLevel.INFO,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name=name or "WebhookNotifier",
            min_level=min_level,
            enabled=enabled,
        )
        self.webhook_url = webhook_url
        self.timeout = timeout

    def _build_payload(self, message: NotificationMessage) -> dict[str, Any]:
        """构建请求 payload（子类应覆盖）

        Parameters
        ----------
        message : NotificationMessage
            通知消息

        Returns
        -------
        dict[str, Any]
            JSON payload
        """
        return {
            "title": message.title,
            "content": message.format_text(),
            "level": message.level.value,
            "timestamp": message.timestamp.isoformat(),
        }

    def _get_url(self) -> str:
        """获取请求 URL（子类可覆盖，如需签名参数）"""
        return self.webhook_url

    def _post_json(self, url: str, payload: dict[str, Any]) -> bool:
        """发送 JSON POST 请求

        使用 urllib 避免引入 requests 依赖。

        Parameters
        ----------
        url : str
            请求地址
        payload : dict
            JSON 数据

        Returns
        -------
        bool
            是否成功（HTTP 200 且无错误码）
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                resp_data = json.loads(resp_body)
                # 检查各平台的通用错误码
                errcode = resp_data.get("errcode", resp_data.get("code", 0))
                if errcode != 0:
                    errmsg = resp_data.get(
                        "errmsg", resp_data.get("msg", resp_body)
                    )
                    logger.error(
                        "Webhook 返回错误 [%s]: code=%s, msg=%s",
                        self.name, errcode, errmsg,
                    )
                    return False
                return True
        except Exception as exc:
            logger.error("Webhook 请求失败 [%s]: %s", self.name, exc)
            return False

    def send(self, message: NotificationMessage) -> bool:
        """发送 Webhook 通知

        Parameters
        ----------
        message : NotificationMessage
            通知消息

        Returns
        -------
        bool
            是否发送成功
        """
        url = self._get_url()
        payload = self._build_payload(message)
        return self._post_json(url, payload)

    def is_available(self) -> bool:
        """检查 Webhook 是否可用"""
        return self.enabled and bool(self.webhook_url)

    def __repr__(self) -> str:
        # 隐藏 URL 中的敏感 token
        url_display = self.webhook_url[:40] + "..." if len(self.webhook_url) > 40 else self.webhook_url
        return (
            f"{self.__class__.__name__}(url={url_display!r}, "
            f"enabled={self.enabled})"
        )


class DingTalkNotifier(WebhookNotifier):
    """钉钉机器人 Webhook 通知

    支持:
    - Markdown 格式消息
    - 加签安全设置（secret）
    - @指定人员

    Parameters
    ----------
    webhook_url : str
        钉钉机器人 Webhook URL
    secret : str
        加签密钥（机器人安全设置中获取），可选
    at_mobiles : list[str]
        需要 @ 的手机号列表
    at_all : bool
        是否 @所有人
    """

    def __init__(
        self,
        webhook_url: str,
        secret: str = "",
        at_mobiles: list[str] | None = None,
        at_all: bool = False,
        timeout: int = 10,
        min_level: NotificationLevel = NotificationLevel.INFO,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            webhook_url=webhook_url,
            timeout=timeout,
            name="DingTalkNotifier",
            min_level=min_level,
            enabled=enabled,
        )
        self.secret = secret
        self.at_mobiles = at_mobiles or []
        self.at_all = at_all

    def _get_url(self) -> str:
        """生成带签名的 URL"""
        if not self.secret:
            return self.webhook_url

        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        separator = "&" if "?" in self.webhook_url else "?"
        return f"{self.webhook_url}{separator}timestamp={timestamp}&sign={sign}"

    def _build_payload(self, message: NotificationMessage) -> dict[str, Any]:
        """构建钉钉消息 payload（Markdown 格式）"""
        md_content = message.format_markdown()

        # 添加 @人员
        if self.at_mobiles:
            mentions = " ".join(f"@{m}" for m in self.at_mobiles)
            md_content += f"\n\n{mentions}"

        return {
            "msgtype": "markdown",
            "markdown": {
                "title": message.title,
                "text": md_content,
            },
            "at": {
                "atMobiles": self.at_mobiles,
                "isAtAll": self.at_all,
            },
        }


class WeChatWorkNotifier(WebhookNotifier):
    """企业微信机器人 Webhook 通知

    支持 Markdown 格式消息。

    Parameters
    ----------
    webhook_url : str
        企业微信机器人 Webhook URL
    mentioned_list : list[str]
        需要 @ 的用户 ID 列表（如 ["user1", "@all"]）
    mentioned_mobile_list : list[str]
        需要 @ 的手机号列表
    """

    def __init__(
        self,
        webhook_url: str,
        mentioned_list: list[str] | None = None,
        mentioned_mobile_list: list[str] | None = None,
        timeout: int = 10,
        min_level: NotificationLevel = NotificationLevel.INFO,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            webhook_url=webhook_url,
            timeout=timeout,
            name="WeChatWorkNotifier",
            min_level=min_level,
            enabled=enabled,
        )
        self.mentioned_list = mentioned_list or []
        self.mentioned_mobile_list = mentioned_mobile_list or []

    def _build_payload(self, message: NotificationMessage) -> dict[str, Any]:
        """构建企业微信消息 payload（Markdown 格式）"""
        md_content = message.format_markdown()

        # 企业微信 markdown 不支持 mentioned，需要用 text 类型
        # 但 markdown 更美观，这里优先用 markdown
        if self.mentioned_list or self.mentioned_mobile_list:
            # 企业微信 markdown 中 @ 不生效，退回 text 类型
            return {
                "msgtype": "text",
                "text": {
                    "content": message.format_text(),
                    "mentioned_list": self.mentioned_list,
                    "mentioned_mobile_list": self.mentioned_mobile_list,
                },
            }

        return {
            "msgtype": "markdown",
            "markdown": {
                "content": md_content,
            },
        }


class SlackNotifier(WebhookNotifier):
    """Slack Incoming Webhook 通知

    使用 Slack Block Kit 格式化消息。

    Parameters
    ----------
    webhook_url : str
        Slack Webhook URL
    channel : str
        频道名称（可选，覆盖 Webhook 默认频道）
    username : str
        机器人显示名称
    icon_emoji : str
        机器人头像 emoji
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str = "",
        username: str = "量化交易系统",
        icon_emoji: str = ":chart_with_upwards_trend:",
        timeout: int = 10,
        min_level: NotificationLevel = NotificationLevel.INFO,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            webhook_url=webhook_url,
            timeout=timeout,
            name="SlackNotifier",
            min_level=min_level,
            enabled=enabled,
        )
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji

    def _build_payload(self, message: NotificationMessage) -> dict[str, Any]:
        """构建 Slack 消息 payload"""
        level_emoji = {
            NotificationLevel.DEBUG: ":mag:",
            NotificationLevel.INFO: ":information_source:",
            NotificationLevel.WARNING: ":warning:",
            NotificationLevel.ERROR: ":x:",
            NotificationLevel.CRITICAL: ":rotating_light:",
        }
        emoji = level_emoji.get(message.level, ":information_source:")
        timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        text = f"{emoji} *{message.title}*\n_{message.level.value.upper()} | {timestamp}_"
        if message.content:
            text += f"\n\n{message.content}"
        if message.extra:
            extra_lines = "\n".join(f"• *{k}:* {v}" for k, v in message.extra.items())
            text += f"\n\n{extra_lines}"

        payload: dict[str, Any] = {
            "text": text,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
        }
        if self.channel:
            payload["channel"] = self.channel

        return payload

    def _post_json(self, url: str, payload: dict[str, Any]) -> bool:
        """Slack 返回 "ok" 而非 JSON errcode"""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                if resp_body.strip() == "ok" or resp.status == 200:
                    return True
                logger.error("Slack 返回错误: %s", resp_body)
                return False
        except Exception as exc:
            logger.error("Slack Webhook 请求失败: %s", exc)
            return False
