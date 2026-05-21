"""Database setup script for the Personal Health Assistant.

Creates MySQL database and tables if they don't exist.
Run: python setup_database.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import mysql.connector
except ImportError:
    logger.error("mysql-connector-python not installed. Run: pip install mysql-connector-python")
    sys.exit(1)

# Configuration
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_ROOT_USER = os.getenv("MYSQL_ROOT_USER", "root")
MYSQL_ROOT_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "health_assistant")
MYSQL_USER = os.getenv("MYSQL_USER", "health_assistant")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "health_assistant_pass")


def create_database():
    """Create the database if it doesn't exist."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_ROOT_USER,
            password=MYSQL_ROOT_PASSWORD,
        )
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        logger.info(f"✅ Database '{MYSQL_DATABASE}' ensured")
        # Create app user
        cursor.execute(
            f"CREATE USER IF NOT EXISTS '{MYSQL_USER}'@'%' IDENTIFIED BY '{MYSQL_PASSWORD}'"
        )
        cursor.execute(
            f"GRANT ALL PRIVILEGES ON {MYSQL_DATABASE}.* TO '{MYSQL_USER}'@'%'"
        )
        cursor.execute("FLUSH PRIVILEGES")
        logger.info(f"✅ User '{MYSQL_USER}' ensured with privileges")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Database creation failed: {e}")
        logger.info("Skipping user creation — database may already exist")


def create_tables():
    """Create all tables using MySQLClient which handles table creation."""
    sys.path.insert(0, os.path.dirname(__file__))
    from customer_support_chat.app.core.mysql_client import MySQLClient
    try:
        client = MySQLClient()
        logger.info("✅ All tables verified/created")
        # Insert sample data
        insert_sample_data(client)
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")


def insert_sample_data(client):
    """Insert sample users and data for development."""
    try:
        # Sample user
        client.execute(
            "INSERT IGNORE INTO users (id, name, email, phone, date_of_birth, gender, blood_type, allergies, chronic_conditions, emergency_contact) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            ("user_001", "Zhang Wei", "zhangwei@example.com", "13800138000",
             "1985-06-15", "male", "A+", "Penicillin, Peanuts",
             "Hypertension (diagnosed 2020)", "Li Mei (wife): 13900139000"),
            fetch=False,
        )
        logger.info("✅ Sample user data inserted")
    except Exception as e:
        logger.warning(f"Sample data: {e}")


if __name__ == "__main__":
    logger.info("🏥 Setting up Personal Health Assistant database...")
    create_database()
    create_tables()
    logger.info("✅ Database setup complete!")
