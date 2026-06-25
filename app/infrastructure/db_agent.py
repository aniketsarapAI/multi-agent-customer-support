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
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        conn.commit()
        return rows
    finally:
        conn.close()
