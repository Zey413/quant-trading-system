"""数据库连接管理 - SQLite 连接池与生命周期管理

特性:
- WAL 模式提升并发读写性能
- 线程安全 (threading.Lock + check_same_thread=False)
- Context Manager 支持
- 自动建表与迁移
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from quant_trading.database.migrations import MigrationManager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 数据库连接管理器

    使用方式::

        # 方式一: Context Manager
        with DatabaseManager("quant.db") as db:
            db.execute("SELECT 1")

        # 方式二: 手动管理
        db = DatabaseManager("quant.db")
        db.connect()
        db.execute("SELECT 1")
        db.close()

        # 方式三: 内存数据库 (测试用)
        db = DatabaseManager(":memory:")
    """

    def __init__(self, db_path: str | Path = "quant.db") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def connection(self) -> sqlite3.Connection:
        """获取当前连接，未连接时自动连接"""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def connect(self) -> None:
        """建立数据库连接并初始化"""
        with self._lock:
            if self._conn is not None:
                return

            # 如果不是内存数据库，确保目录存在
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )

            # 基本配置
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")

            # 自动建表与迁移
            self._init_schema()

            logger.info("数据库连接已建立: %s", self._db_path)

    def _init_schema(self) -> None:
        """初始化表结构 (通过迁移系统)"""
        assert self._conn is not None
        migration_mgr = MigrationManager(self._conn)
        migration_mgr.run_all()

    def close(self) -> None:
        """关闭数据库连接"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                logger.info("数据库连接已关闭: %s", self._db_path)

    def __enter__(self) -> DatabaseManager:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------

    def execute(
        self, sql: str, params: tuple | dict | None = None
    ) -> sqlite3.Cursor:
        """执行单条 SQL (线程安全)"""
        with self._lock:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor

    def executemany(
        self, sql: str, params_list: list[tuple | dict]
    ) -> sqlite3.Cursor:
        """批量执行 SQL (线程安全)"""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.executemany(sql, params_list)
            return cursor

    def executescript(self, script: str) -> None:
        """执行 SQL 脚本 (线程安全)"""
        with self._lock:
            self.connection.executescript(script)

    def fetchone(
        self, sql: str, params: tuple | dict | None = None
    ) -> tuple | None:
        """查询单行"""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(
        self, sql: str, params: tuple | dict | None = None
    ) -> list[tuple]:
        """查询所有行"""
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def fetchval(
        self, sql: str, params: tuple | dict | None = None
    ) -> Any:
        """查询单个值"""
        row = self.fetchone(sql, params)
        return row[0] if row else None

    def commit(self) -> None:
        """提交事务 (线程安全)"""
        with self._lock:
            if self._conn:
                self._conn.commit()

    def rollback(self) -> None:
        """回滚事务 (线程安全)"""
        with self._lock:
            if self._conn:
                self._conn.rollback()

    # ------------------------------------------------------------------
    # 事务上下文管理
    # ------------------------------------------------------------------

    class _Transaction:
        """内部事务管理器"""

        def __init__(self, db: DatabaseManager) -> None:
            self._db = db

        def __enter__(self) -> DatabaseManager._Transaction:
            self._db.execute("BEGIN")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            if exc_type:
                self._db.rollback()
            else:
                self._db.commit()

    def transaction(self) -> _Transaction:
        """获取事务上下文管理器

        使用方式::

            with db.transaction():
                db.execute("INSERT ...")
                db.execute("UPDATE ...")
        """
        return self._Transaction(self)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        row = self.fetchone(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (table_name,),
        )
        return bool(row and row[0] > 0)

    def get_table_names(self) -> list[str]:
        """获取所有表名"""
        rows = self.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [r[0] for r in rows]

    def get_row_count(self, table_name: str) -> int:
        """获取表行数"""
        val = self.fetchval(f"SELECT COUNT(*) FROM [{table_name}]")
        return val if val else 0

    def vacuum(self) -> None:
        """压缩数据库"""
        with self._lock:
            if self._conn:
                self._conn.execute("VACUUM")
                logger.info("数据库已压缩: %s", self._db_path)

    def get_db_size(self) -> int:
        """获取数据库文件大小 (字节), 内存数据库返回 0"""
        if self._db_path == ":memory:":
            return 0
        p = Path(self._db_path)
        return p.stat().st_size if p.exists() else 0
