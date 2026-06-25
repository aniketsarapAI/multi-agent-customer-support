import html
import smtplib
import logging
from datetime import datetime, UTC
from email.message import EmailMessage

from app.config import settings

SEVERITY_MAP = {
    "complaint": ("HIGH", "#dc2626"),
    "frustration": ("MEDIUM", "#d97706"),
    "repeated_negative_sentiment": ("MEDIUM", "#d97706"),
    "unresolved_issue": ("LOW", "#ca8a04"),
    "human_requested": ("INFO", "#2563eb"),
}

ACTION_MAP = {
    "frustration": "Review customer concerns and continue the conversation.",
    "complaint": "Review complaint details and determine the next resolution step.",
    "human_requested": "Customer explicitly requested human assistance. Follow up directly.",
    "unresolved_issue": "Review previous responses and investigate the unresolved issue further.",
    "repeated_negative_sentiment": "Review conversation history for recurring themes and address proactively.",
}


def _severity_label(reason: str) -> str:
    return SEVERITY_MAP.get(reason, ("INFO", "#6b7280"))[0]


def _severity_color(reason: str) -> str:
    return SEVERITY_MAP.get(reason, ("INFO", "#6b7280"))[1]


def _recommended_action(reason: str) -> str:
    return ACTION_MAP.get(reason, "Review the escalation and respond to the customer.")


def _build_html(
    summary: str,
    reason: str,
    docs: list[str],
    sql_queries: list[str],
    message_count: int,
    ticket_id: str,
) -> str:
    severity = _severity_label(reason)
    color = _severity_color(reason)
    action = _recommended_action(reason)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    docs_deduped = list(dict.fromkeys(docs))
    sql_truncated = [q[:300] + "..." if len(q) > 300 else q for q in sql_queries[-5:]]
    metrics_docs = len(set(docs_deduped))
    metrics_sql = len(sql_queries)

    def card(title: str, body: str) -> str:
        escaped = html.escape(body).replace("\n", "<br>")
        return (
            f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:12px;">'
            f'<h3 style="margin:0 0 6px;font-size:14px;color:#374151;">{title}</h3>'
            f'<div style="font-size:13px;color:#4b5563;line-height:1.5;">{escaped}</div></div>'
        )

    # Parse summary sections by known headings
    doc_html = ""
    if docs_deduped:
        items = "".join(f"<li>{html.escape(d)}</li>" for d in docs_deduped)
        doc_html = f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:12px;">'
        f'<h3 style="margin:0 0 6px;font-size:14px;color:#374151;">Documents Consulted</h3>'
        f'<ul style="margin:0;font-size:13px;color:#4b5563;">{items}</ul></div>'

    sql_html = ""
    if sql_truncated:
        blocks = "".join(
            f'<pre style="background:#1f2937;color:#e5e7eb;padding:8px;border-radius:4px;font-size:12px;overflow-x:auto;margin:4px 0;">{html.escape(q)}</pre>'
            for q in sql_truncated
        )
        sql_html = f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:12px;">'
        f'<h3 style="margin:0 0 6px;font-size:14px;color:#374151;">SQL Queries Executed</h3>{blocks}</div>'

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f3f4f6;">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:24px 12px;">
<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);">

<tr><td style="padding:24px 24px 0;">
<h1 style="margin:0;font-size:20px;color:#111827;">🚨 Human Handoff Required</h1>
<p style="margin:4px 0 0;font-size:13px;color:#6b7280;">Ticket: ESC-{html.escape(ticket_id)} · <span style="font-weight:600;color:{color};">{severity}</span></p>
</td></tr>

<tr><td style="padding:12px 24px;">
<table width="100%" cellpadding="6" cellspacing="0" style="font-size:13px;color:#374151;">
<tr><td style="border-bottom:1px solid #e5e7eb;font-weight:600;width:120px;">Reason</td><td style="border-bottom:1px solid #e5e7eb;">{html.escape(reason)}</td></tr>
<tr><td style="border-bottom:1px solid #e5e7eb;font-weight:600;">Severity</td><td style="border-bottom:1px solid #e5e7eb;color:{color};font-weight:600;">{severity}</td></tr>
<tr><td style="border-bottom:1px solid #e5e7eb;font-weight:600;">Messages</td><td style="border-bottom:1px solid #e5e7eb;">{message_count}</td></tr>
<tr><td style="font-weight:600;">Timestamp</td><td>{timestamp}</td></tr>
</table>
</td></tr>

<tr><td style="padding:0 24px 12px;">

{card("User Goal", summary.split("System Actions")[0].replace("User Goal", "").strip().lstrip("-").strip() if "System Actions" in summary else summary[:300])}

{card("Escalation Reason", reason)}

{doc_html}

{sql_html}

<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:12px;">
<table width="100%" cellpadding="4" cellspacing="0" style="font-size:13px;color:#374151;">
<tr><td style="font-weight:600;">Messages</td><td style="text-align:right;">{message_count}</td></tr>
<tr><td style="font-weight:600;">Documents Consulted</td><td style="text-align:right;">{metrics_docs}</td></tr>
<tr><td style="font-weight:600;">SQL Queries Executed</td><td style="text-align:right;">{metrics_sql}</td></tr>
</table>
</div>

{card("Recommended Action", action)}

</td></tr>

<tr><td style="padding:12px 24px 24px;font-size:11px;color:#9ca3af;text-align:center;">
This escalation was automatically generated by Self-RAG MCP.
</td></tr>

</table>
</td></tr></table>
</body>
</html>"""


def send_escalation_email(
    handoff_summary: str,
    reason: str,
    rag_docs_used: list[str] | None = None,
    sql_queries_executed: list[str] | None = None,
    message_count: int = 0,
) -> None:
    user = settings.gmail_user
    password = settings.gmail_app_password
    to = settings.support_email or user
    if not user or not password:
        logging.warning("Gmail not configured — skipping escalation email")
        return

    ticket_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    severity = _severity_label(reason)
    subject = f"[ESC-{ticket_id}] {severity} {reason}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(handoff_summary)

    html_body = _build_html(
        handoff_summary,
        reason,
        rag_docs_used or [],
        sql_queries_executed or [],
        message_count,
        ticket_id,
    )
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
