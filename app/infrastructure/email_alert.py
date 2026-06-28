import datetime
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import settings

logger = logging.getLogger(__name__)


def send_escalation_email(
    handoff_summary: str,
    reason: str,
    *,
    rag_docs_used: list[str] | None = None,
    sql_queries_executed: list[str] | None = None,
    message_count: int = 0,
) -> None:
    sender = settings.gmail_user
    password = settings.gmail_app_password
    recipient = settings.support_email
    if not all([sender, password, recipient]):
        logger.warning("Email settings incomplete — skipping escalation email")
        return

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    severity = "🟠" if reason in ("unresolved_issue",) else "🔴"
    ticket_id = f"ESC-{now_utc[:10].replace('-', '')}-{datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')}"

    docs_section = ""
    if rag_docs_used:
        docs_lines = "\n".join(f"• {d}" for d in rag_docs_used)
        docs_section = f"<h3>📄 Documents Consulted</h3><pre style=\"font-family:monospace;background:#f5f5f5;padding:10px;border-radius:4px;\">{docs_lines}</pre>"

    sql_section = ""
    if sql_queries_executed:
        sql_lines = "\n".join(f"• {q}" for q in sql_queries_executed)
        sql_section = f"<h3>🗄️ SQL Queries Executed</h3><pre style=\"font-family:monospace;background:#f5f5f5;padding:10px;border-radius:4px;\">{sql_lines}</pre>"

    html = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:640px;margin:0 auto;padding:20px;">
<div style="border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
<div style="background:#1a1a2e;color:white;padding:16px 24px;">
<h2 style="margin:0;">{severity} Escalation Alert</h2>
</div>
<div style="padding:24px;">
<table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
<tr><td style="padding:6px 0;color:#666;width:100px;">Ticket ID</td><td style="padding:6px 0;font-weight:600;">{ticket_id}</td></tr>
<tr><td style="padding:6px 0;color:#666;">Severity</td><td style="padding:6px 0;">{reason.replace('_', ' ').title()}</td></tr>
<tr><td style="padding:6px 0;color:#666;">Messages</td><td style="padding:6px 0;">{message_count}</td></tr>
<tr><td style="padding:6px 0;color:#666;">Time</td><td style="padding:6px 0;">{now_utc}</td></tr>
</table>
<h3>📋 Handoff Summary</h3>
<pre style="font-family:monospace;background:#f5f5f5;padding:10px;border-radius:4px;white-space:pre-wrap;">{handoff_summary}</pre>
{docs_section}
{sql_section}
</div>
</div>
<p style="font-size:12px;color:#999;text-align:center;margin-top:16px;">Automated escalation from Self-RAG Customer Support System</p>
</body>
</html>"""

    text = f"Escalation Ticket: {ticket_id}\nSeverity: {reason}\n\n{handoff_summary}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{severity} [Escalation] {ticket_id} — {reason.replace('_', ' ').title()}"
    msg["From"] = sender
    msg["To"] = recipient

    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        logger.info(f"Escalation email sent: {ticket_id}")
    except Exception:
        logger.exception("Failed to send escalation email")
        raise
