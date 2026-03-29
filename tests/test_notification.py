"""通知模块单元测试

测试覆盖:
- NotificationMessage 模型与格式化
- Notifier 基类行为（过滤、计数、安全发送）
- EmailNotifier SMTP 发送（mock）
- DingTalkNotifier / WeChatWorkNotifier / SlackNotifier Webhook 发送（mock）
- NotificationManager 多渠道管理、统一发送、历史记录
- 不依赖真实网络调用（全部 mock）
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest

from quant_trading.notification.base import (
    Notifier,
    NotificationLevel,
    NotificationMessage,
)
from quant_trading.notification.email_notifier import EmailNotifier
from quant_trading.notification.webhook_notifier import (
    DingTalkNotifier,
    WeChatWorkNotifier,
    SlackNotifier,
    WebhookNotifier,
)
from quant_trading.notification.manager import NotificationManager


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def sample_message() -> NotificationMessage:
    """标准测试消息"""
    return NotificationMessage(
        title="测试通知",
        content="这是一条测试内容",
        level=NotificationLevel.INFO,
    )


@pytest.fixture()
def warning_message() -> NotificationMessage:
    """警告级别测试消息"""
    return NotificationMessage(
        title="风险预警",
        content="000001 跌幅超过5%",
        level=NotificationLevel.WARNING,
        extra={"symbol": "000001", "pct_change": -5.2},
    )


@pytest.fixture()
def error_message() -> NotificationMessage:
    """错误级别测试消息"""
    return NotificationMessage(
        title="系统错误",
        content="数据获取失败",
        level=NotificationLevel.ERROR,
    )


class FakeNotifier(Notifier):
    """测试用的假通知渠道"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sent_messages: list[NotificationMessage] = []
        self.should_fail = False

    def send(self, message: NotificationMessage) -> bool:
        if self.should_fail:
            raise RuntimeError("模拟发送失败")
        self.sent_messages.append(message)
        return True


# ======================================================================
# Tests: NotificationMessage
# ======================================================================


class TestNotificationMessage:
    """通知消息模型测试"""

    def test_create_basic_message(self):
        """创建基本消息"""
        msg = NotificationMessage(title="测试")
        assert msg.title == "测试"
        assert msg.content == ""
        assert msg.level == NotificationLevel.INFO
        assert isinstance(msg.timestamp, datetime)

    def test_create_full_message(self, warning_message):
        """创建完整消息"""
        assert warning_message.title == "风险预警"
        assert warning_message.level == NotificationLevel.WARNING
        assert "symbol" in warning_message.extra
        assert warning_message.extra["symbol"] == "000001"

    def test_format_text(self, sample_message):
        """纯文本格式化"""
        text = sample_message.format_text()
        assert "[INFO]" in text
        assert "测试通知" in text
        assert "测试内容" in text

    def test_format_markdown(self, warning_message):
        """Markdown 格式化"""
        md = warning_message.format_markdown()
        assert "风险预警" in md
        assert "WARNING" in md
        assert "symbol" in md

    def test_level_values(self):
        """通知级别枚举"""
        assert NotificationLevel.DEBUG.value == "debug"
        assert NotificationLevel.INFO.value == "info"
        assert NotificationLevel.WARNING.value == "warning"
        assert NotificationLevel.ERROR.value == "error"
        assert NotificationLevel.CRITICAL.value == "critical"


# ======================================================================
# Tests: Notifier Base
# ======================================================================


class TestNotifierBase:
    """Notifier 基类测试"""

    def test_fake_notifier_send(self, sample_message):
        """基本发送"""
        notifier = FakeNotifier(name="test")
        assert notifier.send(sample_message) is True
        assert len(notifier.sent_messages) == 1

    def test_safe_send_success(self, sample_message):
        """safe_send 成功时计数"""
        notifier = FakeNotifier(name="test")
        result = notifier.safe_send(sample_message)
        assert result is True
        assert notifier.send_count == 1
        assert notifier.error_count == 0

    def test_safe_send_failure(self, sample_message):
        """safe_send 失败时计数"""
        notifier = FakeNotifier(name="test")
        notifier.should_fail = True
        result = notifier.safe_send(sample_message)
        assert result is False
        assert notifier.send_count == 0
        assert notifier.error_count == 1

    def test_level_filter(self):
        """级别过滤"""
        notifier = FakeNotifier(
            name="test", min_level=NotificationLevel.WARNING
        )
        # INFO 消息应被过滤
        info_msg = NotificationMessage(title="info", level=NotificationLevel.INFO)
        result = notifier.safe_send(info_msg)
        assert result is False
        assert len(notifier.sent_messages) == 0

        # WARNING 消息应通过
        warn_msg = NotificationMessage(
            title="warn", level=NotificationLevel.WARNING
        )
        result = notifier.safe_send(warn_msg)
        assert result is True
        assert len(notifier.sent_messages) == 1

    def test_disabled_notifier(self, sample_message):
        """禁用的通知渠道不发送"""
        notifier = FakeNotifier(name="test", enabled=False)
        result = notifier.safe_send(sample_message)
        assert result is False

    def test_should_send_level_ordering(self):
        """should_send 级别顺序判断"""
        notifier = FakeNotifier(
            name="test", min_level=NotificationLevel.ERROR
        )

        # DEBUG -> 不发送
        msg_debug = NotificationMessage(
            title="d", level=NotificationLevel.DEBUG
        )
        assert notifier.should_send(msg_debug) is False

        # ERROR -> 发送
        msg_error = NotificationMessage(
            title="e", level=NotificationLevel.ERROR
        )
        assert notifier.should_send(msg_error) is True

        # CRITICAL -> 发送
        msg_critical = NotificationMessage(
            title="c", level=NotificationLevel.CRITICAL
        )
        assert notifier.should_send(msg_critical) is True

    def test_name_property(self):
        """name 属性"""
        n = FakeNotifier(name="my_channel")
        assert n.name == "my_channel"

    def test_default_name(self):
        """默认 name 为类名"""
        n = FakeNotifier()
        assert n.name == "FakeNotifier"

    def test_repr(self):
        """repr 格式"""
        n = FakeNotifier(name="test")
        r = repr(n)
        assert "FakeNotifier" in r
        assert "test" in r

    def test_is_available(self):
        """is_available 默认返回 enabled"""
        n1 = FakeNotifier(enabled=True)
        assert n1.is_available() is True
        n2 = FakeNotifier(enabled=False)
        assert n2.is_available() is False


# ======================================================================
# Tests: EmailNotifier
# ======================================================================


class TestEmailNotifier:
    """EmailNotifier 测试"""

    def test_create_notifier(self):
        """创建邮件通知"""
        notifier = EmailNotifier(
            smtp_host="smtp.qq.com",
            smtp_port=465,
            username="test@qq.com",
            password="test_pass",
            recipients=["target@example.com"],
        )
        assert notifier.smtp_host == "smtp.qq.com"
        assert notifier.smtp_port == 465
        assert len(notifier.recipients) == 1

    def test_send_success(self, sample_message):
        """成功发送邮件"""
        notifier = EmailNotifier(
            smtp_host="smtp.test.com",
            smtp_port=465,
            username="test@test.com",
            password="pass",
            recipients=["r@test.com"],
        )
        with patch.object(notifier, "_get_smtp_connection") as mock_conn:
            mock_server = MagicMock()
            mock_conn.return_value = mock_server

            result = notifier.send(sample_message)

            assert result is True
            mock_server.sendmail.assert_called_once()
            mock_server.quit.assert_called_once()

    def test_send_empty_recipients(self, sample_message):
        """空收件人列表应返回 False"""
        notifier = EmailNotifier(
            smtp_host="smtp.test.com",
            recipients=[],
        )
        result = notifier.send(sample_message)
        assert result is False

    def test_send_auth_failure(self, sample_message):
        """SMTP 认证失败"""
        import smtplib

        notifier = EmailNotifier(
            smtp_host="smtp.test.com",
            smtp_port=465,
            username="test@test.com",
            password="wrong",
            recipients=["r@test.com"],
        )
        with patch.object(notifier, "_get_smtp_connection") as mock_conn:
            mock_conn.side_effect = smtplib.SMTPAuthenticationError(
                535, b"Auth failed"
            )
            result = notifier.send(sample_message)
            assert result is False

    def test_create_message_html(self, warning_message):
        """邮件消息包含 HTML"""
        notifier = EmailNotifier(
            smtp_host="smtp.test.com",
            username="test@test.com",
            recipients=["r@test.com"],
        )
        mime_msg = notifier._create_message(warning_message)
        assert mime_msg["Subject"] is not None
        assert "WARNING" in mime_msg["Subject"]

    def test_is_available(self):
        """is_available 检查"""
        n1 = EmailNotifier(
            smtp_host="smtp.test.com",
            recipients=["r@test.com"],
        )
        assert n1.is_available() is True

        n2 = EmailNotifier(smtp_host="", recipients=[])
        assert n2.is_available() is False

    def test_repr(self):
        """repr 格式"""
        n = EmailNotifier(
            smtp_host="smtp.test.com",
            smtp_port=465,
            recipients=["a@b.com", "c@d.com"],
        )
        r = repr(n)
        assert "EmailNotifier" in r
        assert "smtp.test.com" in r


# ======================================================================
# Tests: Webhook Notifiers
# ======================================================================


class TestWebhookNotifier:
    """WebhookNotifier 基类测试"""

    def test_build_payload(self, sample_message):
        """构建 payload"""
        notifier = WebhookNotifier(webhook_url="https://example.com/hook")
        payload = notifier._build_payload(sample_message)
        assert "title" in payload
        assert payload["title"] == "测试通知"
        assert "level" in payload

    def test_is_available(self):
        """可用性检查"""
        n1 = WebhookNotifier(webhook_url="https://example.com/hook")
        assert n1.is_available() is True

        n2 = WebhookNotifier(webhook_url="")
        assert n2.is_available() is False

    def test_send_with_mock(self, sample_message):
        """mock HTTP 调用"""
        notifier = WebhookNotifier(webhook_url="https://example.com/hook")
        with patch.object(notifier, "_post_json", return_value=True) as mock:
            result = notifier.send(sample_message)
            assert result is True
            mock.assert_called_once()

    def test_repr_hides_url(self):
        """repr 应截断 URL"""
        long_url = "https://example.com/very/long/webhook/url/with/token/abcdefghijklmnop"
        n = WebhookNotifier(webhook_url=long_url)
        r = repr(n)
        assert "..." in r


class TestDingTalkNotifier:
    """钉钉通知测试"""

    def test_build_payload(self, sample_message):
        """构建钉钉 Markdown payload"""
        notifier = DingTalkNotifier(
            webhook_url="https://oapi.dingtalk.com/robot/send?access_token=xxx"
        )
        payload = notifier._build_payload(sample_message)
        assert payload["msgtype"] == "markdown"
        assert "title" in payload["markdown"]
        assert "text" in payload["markdown"]

    def test_build_payload_with_at(self, sample_message):
        """构建带 @功能 的 payload"""
        notifier = DingTalkNotifier(
            webhook_url="https://oapi.dingtalk.com/robot/send?access_token=xxx",
            at_mobiles=["13800138000"],
            at_all=False,
        )
        payload = notifier._build_payload(sample_message)
        assert payload["at"]["atMobiles"] == ["13800138000"]
        assert payload["at"]["isAtAll"] is False

    def test_get_url_no_secret(self):
        """无密钥时 URL 不变"""
        url = "https://oapi.dingtalk.com/robot/send?access_token=xxx"
        notifier = DingTalkNotifier(webhook_url=url)
        assert notifier._get_url() == url

    def test_get_url_with_secret(self):
        """有密钥时 URL 带签名"""
        notifier = DingTalkNotifier(
            webhook_url="https://oapi.dingtalk.com/robot/send?access_token=xxx",
            secret="SEC123456",
        )
        signed_url = notifier._get_url()
        assert "timestamp=" in signed_url
        assert "sign=" in signed_url

    def test_send_with_mock(self, sample_message):
        """mock 发送"""
        notifier = DingTalkNotifier(
            webhook_url="https://oapi.dingtalk.com/robot/send?access_token=xxx"
        )
        with patch.object(notifier, "_post_json", return_value=True):
            result = notifier.send(sample_message)
            assert result is True


class TestWeChatWorkNotifier:
    """企业微信通知测试"""

    def test_build_markdown_payload(self, sample_message):
        """无 @人员时使用 Markdown"""
        notifier = WeChatWorkNotifier(
            webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
        )
        payload = notifier._build_payload(sample_message)
        assert payload["msgtype"] == "markdown"
        assert "content" in payload["markdown"]

    def test_build_text_payload_with_mention(self, sample_message):
        """有 @人员时回退到 text 类型"""
        notifier = WeChatWorkNotifier(
            webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
            mentioned_list=["user1"],
        )
        payload = notifier._build_payload(sample_message)
        assert payload["msgtype"] == "text"
        assert "mentioned_list" in payload["text"]


class TestSlackNotifier:
    """Slack 通知测试"""

    def test_build_payload(self, sample_message):
        """构建 Slack payload"""
        notifier = SlackNotifier(
            webhook_url="https://hooks.slack.com/services/xxx",
        )
        payload = notifier._build_payload(sample_message)
        assert "text" in payload
        assert "测试通知" in payload["text"]
        assert payload["username"] == "量化交易系统"

    def test_build_payload_with_channel(self, sample_message):
        """指定频道"""
        notifier = SlackNotifier(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#trading",
        )
        payload = notifier._build_payload(sample_message)
        assert payload["channel"] == "#trading"


# ======================================================================
# Tests: NotificationManager
# ======================================================================


class TestNotificationManager:
    """通知管理器测试"""

    def test_add_notifier(self):
        """注册渠道"""
        manager = NotificationManager()
        notifier = FakeNotifier(name="fake")
        manager.add_notifier(notifier)
        assert "fake" in manager.notifiers

    def test_add_duplicate_notifier(self):
        """重复注册应抛出 ValueError"""
        manager = NotificationManager()
        manager.add_notifier(FakeNotifier(name="fake"))
        with pytest.raises(ValueError, match="已存在"):
            manager.add_notifier(FakeNotifier(name="fake"))

    def test_remove_notifier(self):
        """移除渠道"""
        manager = NotificationManager()
        manager.add_notifier(FakeNotifier(name="fake"))
        assert manager.remove_notifier("fake") is True
        assert "fake" not in manager.notifiers

    def test_remove_nonexistent(self):
        """移除不存在的渠道"""
        manager = NotificationManager()
        assert manager.remove_notifier("xxx") is False

    def test_get_notifier(self):
        """获取渠道"""
        manager = NotificationManager()
        notifier = FakeNotifier(name="fake")
        manager.add_notifier(notifier)
        assert manager.get_notifier("fake") is notifier
        assert manager.get_notifier("xxx") is None

    def test_send_to_all(self, sample_message):
        """发送到所有渠道"""
        manager = NotificationManager()
        n1 = FakeNotifier(name="ch1")
        n2 = FakeNotifier(name="ch2")
        manager.add_notifier(n1)
        manager.add_notifier(n2)

        results = manager.send(sample_message)

        assert results["ch1"] is True
        assert results["ch2"] is True
        assert len(n1.sent_messages) == 1
        assert len(n2.sent_messages) == 1

    def test_send_to_specific_channels(self, sample_message):
        """发送到指定渠道"""
        manager = NotificationManager()
        n1 = FakeNotifier(name="ch1")
        n2 = FakeNotifier(name="ch2")
        manager.add_notifier(n1)
        manager.add_notifier(n2)

        results = manager.send(sample_message, channels=["ch1"])

        assert "ch1" in results
        assert "ch2" not in results
        assert len(n1.sent_messages) == 1
        assert len(n2.sent_messages) == 0

    def test_notify_convenience(self):
        """notify 便捷方法"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        results = manager.notify("标题", "内容")
        assert results["ch1"] is True
        assert n.sent_messages[0].title == "标题"

    def test_alert_convenience(self):
        """alert 便捷方法使用 WARNING 级别"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        manager.alert("告警")
        assert n.sent_messages[0].level == NotificationLevel.WARNING

    def test_error_convenience(self):
        """error 便捷方法使用 ERROR 级别"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        manager.error("错误")
        assert n.sent_messages[0].level == NotificationLevel.ERROR

    def test_critical_convenience(self):
        """critical 便捷方法使用 CRITICAL 级别"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        manager.critical("严重")
        assert n.sent_messages[0].level == NotificationLevel.CRITICAL

    def test_send_with_extra(self):
        """发送带附加数据的消息"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        manager.notify(
            "信号",
            "买入信号",
            extra={"symbol": "000001", "price": 13.72},
        )
        assert n.sent_messages[0].extra["symbol"] == "000001"

    def test_history(self, sample_message):
        """发送历史记录"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        manager.send(sample_message)
        manager.send(sample_message)

        assert len(manager.history) == 2
        assert manager.history[0]["title"] == "测试通知"

    def test_history_limit(self):
        """历史记录上限"""
        manager = NotificationManager(max_history=5)
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        for i in range(10):
            manager.notify(f"msg_{i}")

        assert len(manager.history) == 5
        assert manager.history[0]["title"] == "msg_5"

    def test_clear_history(self, sample_message):
        """清空历史"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)
        manager.send(sample_message)
        manager.clear_history()
        assert len(manager.history) == 0

    def test_get_status(self):
        """获取状态"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        status = manager.get_status()
        assert "ch1" in status
        assert status["ch1"]["enabled"] is True
        assert status["ch1"]["send_count"] == 0

    def test_send_async(self, sample_message):
        """异步发送"""
        manager = NotificationManager()
        n = FakeNotifier(name="ch1")
        manager.add_notifier(n)

        future = manager.send_async(sample_message)
        results = future.result(timeout=5)

        assert results["ch1"] is True
        manager.shutdown()

    def test_send_with_failure(self, sample_message):
        """部分渠道失败不影响其他渠道"""
        manager = NotificationManager()
        n1 = FakeNotifier(name="ch1")
        n2 = FakeNotifier(name="ch2")
        n2.should_fail = True
        manager.add_notifier(n1)
        manager.add_notifier(n2)

        results = manager.send(sample_message)

        assert results["ch1"] is True
        assert results["ch2"] is False

    def test_repr(self):
        """repr 格式"""
        manager = NotificationManager()
        r = repr(manager)
        assert "NotificationManager" in r
