"""通知推送模块 - 多渠道消息通知

提供统一的通知接口，支持:
- 邮件推送 (SMTP)
- 钉钉 Webhook
- 企业微信 Webhook
- Slack Webhook

用法::

    from quant_trading.notification import NotificationManager, EmailNotifier

    manager = NotificationManager()
    manager.add_notifier(EmailNotifier(
        smtp_host="smtp.qq.com",
        smtp_port=465,
        username="user@qq.com",
        password="auth_code",
        recipients=["target@example.com"],
    ))
    manager.notify("交易信号", "000001 平安银行触发买入信号")
"""

from quant_trading.notification.base import Notifier, NotificationLevel
from quant_trading.notification.email_notifier import EmailNotifier
from quant_trading.notification.webhook_notifier import (
    DingTalkNotifier,
    WeChatWorkNotifier,
    SlackNotifier,
    WebhookNotifier,
)
from quant_trading.notification.manager import NotificationManager

__all__ = [
    "Notifier",
    "NotificationLevel",
    "EmailNotifier",
    "DingTalkNotifier",
    "WeChatWorkNotifier",
    "SlackNotifier",
    "WebhookNotifier",
    "NotificationManager",
]
