"""邮件通知 - SMTP邮件推送

支持 SSL/TLS 加密连接，兼容 QQ邮箱、163邮箱、Gmail 等。

用法::

    from quant_trading.notification.email_notifier import EmailNotifier

    notifier = EmailNotifier(
        smtp_host="smtp.qq.com",
        smtp_port=465,
        username="your@qq.com",
        password="your_auth_code",
        recipients=["target@example.com"],
        use_ssl=True,
    )
    # 通过 NotificationManager 或直接调用
    from quant_trading.notification.base import NotificationMessage
    msg = NotificationMessage(title="测试", content="这是一封测试邮件")
    notifier.send(msg)
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from quant_trading.notification.base import (
    Notifier,
    NotificationLevel,
    NotificationMessage,
)

logger = logging.getLogger(__name__)


class EmailNotifier(Notifier):
    """SMTP 邮件通知渠道

    Parameters
    ----------
    smtp_host : str
        SMTP 服务器地址
    smtp_port : int
        SMTP 端口（SSL 通常为 465，STARTTLS 为 587）
    username : str
        发件人邮箱
    password : str
        邮箱密码/授权码
    recipients : list[str]
        收件人列表
    sender_name : str
        发件人显示名称
    use_ssl : bool
        是否使用 SSL（默认 True）
    use_tls : bool
        是否使用 STARTTLS（默认 False，与 SSL 互斥）
    timeout : int
        SMTP 连接超时时间（秒）
    min_level : NotificationLevel
        最低通知级别
    enabled : bool
        是否启用
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 465,
        username: str = "",
        password: str = "",
        recipients: list[str] | None = None,
        sender_name: str = "量化交易系统",
        use_ssl: bool = True,
        use_tls: bool = False,
        timeout: int = 30,
        min_level: NotificationLevel = NotificationLevel.INFO,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="EmailNotifier", min_level=min_level, enabled=enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients or []
        self.sender_name = sender_name
        self.use_ssl = use_ssl
        self.use_tls = use_tls
        self.timeout = timeout

    def _create_message(self, notification: NotificationMessage) -> MIMEMultipart:
        """创建邮件 MIME 消息

        Parameters
        ----------
        notification : NotificationMessage
            通知消息

        Returns
        -------
        MIMEMultipart
            MIME 邮件对象
        """
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self.sender_name} <{self.username}>"
        msg["To"] = ", ".join(self.recipients)
        msg["Subject"] = f"[{notification.level.value.upper()}] {notification.title}"

        # 纯文本版本
        text_body = notification.format_text()
        msg.attach(MIMEText(text_body, "plain", "utf-8"))

        # HTML 版本
        html_body = self._render_html(notification)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        return msg

    def _render_html(self, notification: NotificationMessage) -> str:
        """渲染 HTML 邮件内容"""
        level_colors = {
            NotificationLevel.DEBUG: "#6c757d",
            NotificationLevel.INFO: "#0d6efd",
            NotificationLevel.WARNING: "#ffc107",
            NotificationLevel.ERROR: "#dc3545",
            NotificationLevel.CRITICAL: "#dc3545",
        }
        color = level_colors.get(notification.level, "#0d6efd")
        timestamp = notification.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        extra_html = ""
        if notification.extra:
            extra_rows = "".join(
                f"<tr><td style='padding:4px 8px;font-weight:bold'>{k}</td>"
                f"<td style='padding:4px 8px'>{v}</td></tr>"
                for k, v in notification.extra.items()
            )
            extra_html = (
                f"<h3>附加信息</h3>"
                f"<table border='1' cellspacing='0' style='border-collapse:collapse'>"
                f"{extra_rows}</table>"
            )

        content_html = notification.content.replace("\n", "<br>")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 16px;">
                <h2 style="color: {color}; margin-bottom: 4px;">{notification.title}</h2>
                <p style="color: #666; font-size: 12px;">
                    {notification.level.value.upper()} | {timestamp}
                </p>
            </div>
            <div style="margin-top: 16px; line-height: 1.6;">
                {content_html}
            </div>
            {extra_html}
            <hr style="margin-top: 24px; border: none; border-top: 1px solid #eee;">
            <p style="color: #999; font-size: 11px;">
                此邮件由量化交易系统自动发送，请勿直接回复。
            </p>
        </body>
        </html>
        """

    def _get_smtp_connection(self) -> smtplib.SMTP | smtplib.SMTP_SSL:
        """建立 SMTP 连接

        Returns
        -------
        smtplib.SMTP | smtplib.SMTP_SSL
            SMTP 连接对象
        """
        if self.use_ssl:
            server = smtplib.SMTP_SSL(
                self.smtp_host, self.smtp_port, timeout=self.timeout
            )
        else:
            server = smtplib.SMTP(
                self.smtp_host, self.smtp_port, timeout=self.timeout
            )
            if self.use_tls:
                server.starttls()

        if self.username and self.password:
            server.login(self.username, self.password)

        return server

    def send(self, message: NotificationMessage) -> bool:
        """发送邮件通知

        Parameters
        ----------
        message : NotificationMessage
            通知消息

        Returns
        -------
        bool
            是否发送成功
        """
        if not self.recipients:
            logger.warning("EmailNotifier: 收件人列表为空")
            return False

        try:
            mime_msg = self._create_message(message)
            server = self._get_smtp_connection()
            try:
                server.sendmail(self.username, self.recipients, mime_msg.as_string())
            finally:
                server.quit()

            logger.info(
                "邮件发送成功: '%s' -> %s",
                message.title,
                self.recipients,
            )
            return True

        except smtplib.SMTPAuthenticationError as exc:
            logger.error("SMTP 认证失败: %s", exc)
            return False
        except smtplib.SMTPException as exc:
            logger.error("SMTP 发送失败: %s", exc)
            return False
        except Exception as exc:
            logger.error("邮件发送异常: %s", exc)
            return False

    def is_available(self) -> bool:
        """检查 SMTP 是否可连接"""
        if not self.enabled:
            return False
        if not self.smtp_host or not self.recipients:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"EmailNotifier(host={self.smtp_host!r}, port={self.smtp_port}, "
            f"recipients={len(self.recipients)}, enabled={self.enabled})"
        )
