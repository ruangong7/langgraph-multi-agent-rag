"""
MySQL Database Client — Connection pool and CRUD helpers for the Health Assistant.

Uses mysql-connector-python with connection pooling for production-grade
database access. All health data (users, appointments, medications, records)
is stored in MySQL instead of SQLite.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import mysql.connector
    from mysql.connector import pooling
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("mysql-connector-python not installed. Run: pip install mysql-connector-python")


class MySQLConfig:
    """MySQL connection configuration from environment variables."""

    HOST = os.getenv("MYSQL_HOST", "localhost")
    PORT = int(os.getenv("MYSQL_PORT", "3306"))
    USER = os.getenv("MYSQL_USER", "health_assistant")
    PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    DATABASE = os.getenv("MYSQL_DATABASE", "health_assistant")
    POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "10"))
    POOL_NAME = "health_assistant_pool"


class MySQLClient:
    """Thread-safe MySQL client with connection pooling."""

    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not MYSQL_AVAILABLE:
            raise ImportError("mysql-connector-python is required. Run: pip install mysql-connector-python")
        if self._pool is None:
            self._init_pool()

    def _init_pool(self):
        """Initialize the connection pool."""
        try:
            self._pool = pooling.MySQLConnectionPool(
                pool_name=MySQLConfig.POOL_NAME,
                pool_size=MySQLConfig.POOL_SIZE,
                host=MySQLConfig.HOST,
                port=MySQLConfig.PORT,
                user=MySQLConfig.USER,
                password=MySQLConfig.PASSWORD,
                database=MySQLConfig.DATABASE,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci',
                autocommit=False,
            )
            logger.info(f"🔗 MySQL pool initialized: {MySQLConfig.HOST}:{MySQLConfig.PORT}/{MySQLConfig.DATABASE}")
            self._ensure_tables()
        except Exception as e:
            logger.error(f"❌ MySQL connection failed: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)."""
        conn = self._pool.get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = None, fetch: bool = True) -> Optional[List[Dict[str, Any]]]:
        """Execute a SQL query with parameters."""
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(sql, params or ())
                if fetch and cursor.description:
                    result = cursor.fetchall()
                else:
                    result = None
                conn.commit()
                return result
            except Exception as e:
                conn.rollback()
                logger.error(f"SQL error: {e}\n  SQL: {sql}\n  Params: {params}")
                raise
            finally:
                cursor.close()

    def execute_many(self, sql: str, params_list: List[tuple]) -> int:
        """Execute a batch insert/update."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, params_list)
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                conn.rollback()
                logger.error(f"Batch SQL error: {e}")
                raise
            finally:
                cursor.close()

    def _ensure_tables(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(128) NOT NULL,
                    email VARCHAR(256),
                    phone VARCHAR(32),
                    date_of_birth DATE,
                    gender ENUM('male', 'female', 'other') DEFAULT 'other',
                    blood_type VARCHAR(8),
                    allergies TEXT,
                    chronic_conditions TEXT,
                    emergency_contact VARCHAR(256),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    doctor_name VARCHAR(128),
                    hospital_name VARCHAR(256),
                    department VARCHAR(128),
                    appointment_time DATETIME,
                    status ENUM('scheduled', 'confirmed', 'cancelled', 'completed') DEFAULT 'scheduled',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medications (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    medication_name VARCHAR(256) NOT NULL,
                    dosage VARCHAR(128),
                    frequency VARCHAR(128),
                    start_date DATE,
                    end_date DATE,
                    reminder_time TIME,
                    status ENUM('active', 'paused', 'completed') DEFAULT 'active',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medical_records (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    record_type ENUM('diagnosis', 'lab_result', 'prescription', 'vaccination', 'surgery', 'other'),
                    title VARCHAR(256),
                    description TEXT,
                    doctor_name VARCHAR(128),
                    hospital_name VARCHAR(256),
                    record_date DATE,
                    attachments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_assessments (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    assessment_type VARCHAR(128),
                    symptoms TEXT,
                    risk_level ENUM('low', 'medium', 'high', 'urgent'),
                    recommendation TEXT,
                    score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emergency_contacts (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    contact_name VARCHAR(128),
                    relationship VARCHAR(64),
                    phone VARCHAR(32),
                    is_primary BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            conn.commit()
            cursor.close()
            logger.info("✅ MySQL tables verified/created")

    def health_check(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            result = self.execute("SELECT 1 AS ok", fetch=True)
            return {"status": "healthy", "connected": True, "host": MySQLConfig.HOST, "database": MySQLConfig.DATABASE}
        except Exception as e:
            return {"status": "unhealthy", "connected": False, "error": str(e)}

    @property
    def is_connected(self) -> bool:
        try:
            self.execute("SELECT 1", fetch=False)
            return True
        except Exception:
            return False


# Singleton instance
mysql_client = MySQLClient() if MYSQL_AVAILABLE else None
