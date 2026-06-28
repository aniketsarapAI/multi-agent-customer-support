import pymysql
import pymysql.cursors

from app.config import settings

TABLE_SCHEMA = """
Table: customers (columns: customer_id TEXT, customer_unique_id TEXT, customer_zip_code_prefix TEXT, customer_city TEXT, customer_state TEXT)
Table: sellers (columns: seller_id TEXT, seller_zip_code_prefix TEXT, seller_city TEXT, seller_state TEXT)
Table: products (columns: product_id TEXT, product_category_name TEXT, product_name_lenght TEXT, product_description_lenght TEXT, product_photos_qty TEXT, product_weight_g TEXT, product_length_cm TEXT, product_height_cm TEXT, product_width_cm TEXT)
Table: orders (columns: order_id TEXT, customer_id TEXT, order_status TEXT, order_purchase_timestamp TEXT, order_approved_at TEXT, order_delivered_carrier_date TEXT, order_delivered_customer_date TEXT, order_estimated_delivery_date TEXT)
Table: order_items (columns: order_id TEXT, order_item_id TEXT, product_id TEXT, seller_id TEXT, shipping_limit_date TEXT, price TEXT, freight_value TEXT)
Table: order_payments (columns: order_id TEXT, payment_sequential TEXT, payment_type TEXT, payment_installments TEXT, payment_value TEXT)
Table: order_reviews (columns: review_id TEXT, order_id TEXT, review_score TEXT, review_comment_title TEXT, review_comment_message TEXT, review_creation_date TEXT, review_answer_timestamp TEXT)
Table: geolocation (columns: geolocation_zip_code_prefix TEXT, geolocation_lat TEXT, geolocation_lng TEXT, geolocation_city TEXT, geolocation_state TEXT)
Table: product_category_name_translation (columns: product_category_name TEXT, product_category_name_english TEXT) — Maps Portuguese category names to English (e.g., beleza_saude → health_beauty). Use this when filtering by product category name.
Table: leads_qualified (columns: mql_id TEXT, first_contact_date TEXT, landing_page_id TEXT, origin TEXT)
Table: leads_closed (columns: mql_id TEXT, seller_id TEXT, sdr_id TEXT, sr_id TEXT, won_date TEXT, business_segment TEXT, lead_type TEXT, lead_behaviour_profile TEXT, has_company TEXT, has_gtin TEXT, average_stock TEXT, business_type TEXT, declared_product_catalog_size TEXT, declared_monthly_revenue TEXT)
"""


class SQLValidationError(Exception):
    """Raised when SQL fails read-only validation (e.g. non-SELECT, multi-statement)."""


class SQLSyntaxError(Exception):
    """Raised when the database rejects the SQL as malformed (retryable via regeneration)."""


class SQLConnectionError(Exception):
    """Raised on connection/server issues (not retryable via SQL regeneration)."""


# pymysql error codes that indicate connection/server issues, not SQL syntax problems.
_CONNECTION_ERROR_CODES = {1042, 1045, 1080, 1081, 1082, 1083, 1084, 1090, 1091,
                            1129, 1130, 1153, 1154, 1155, 1156, 1157, 1158, 1159,
                            1184, 1213, 1226, 1243, 1814, 1864, 1881, 1893, 1894,
                            2000, 2001, 2002, 2003, 2005, 2006, 2007, 2009, 2013,
                            2020, 2026, 2027, 2055, 2059, 2061}


def validate_select_only(sql: str) -> None:
    """Reject multi-statement SQL and non-read-only statements.

    Allows only SELECT and WITH (CTE) as leading keywords. Disables stacked
    queries by rejecting any semicolon in the body.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise SQLValidationError("Empty SQL query.")
    if ";" in stripped:
        raise SQLValidationError("Multiple statements are not allowed.")
    first_word = stripped.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        raise SQLValidationError(
            f"Only read-only SELECT/WITH queries are allowed; got '{first_word}'."
        )


def get_connection():
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        ssl={"ssl": True},
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=30,
    )


def execute_sql(sql: str) -> list[dict]:
    validate_select_only(sql)
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        conn.commit()
        return rows
    except pymysql.err.OperationalError as e:
        code = e.args[0] if e.args else 0
        if code in _CONNECTION_ERROR_CODES:
            raise SQLConnectionError(str(e)) from e
        # Other OperationalErrors (e.g. unknown column) are syntax-shaped.
        raise SQLSyntaxError(str(e)) from e
    except pymysql.err.ProgrammingError as e:
        raise SQLSyntaxError(str(e)) from e
    except pymysql.err.InternalError as e:
        raise SQLSyntaxError(str(e)) from e
    except pymysql.err.NotSupportedError as e:
        raise SQLSyntaxError(str(e)) from e
    finally:
        conn.close()
