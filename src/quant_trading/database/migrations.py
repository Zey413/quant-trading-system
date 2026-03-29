"""数据库版本化迁移

迁移策略:
- 每个版本对应一个迁移函数 _migrate_vN
- schema_version 表记录当前版本
- 连接时自动检测并执行未完成的迁移
- 每次迁移在独立事务中执行
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable

logger = logging.getLogger(__name__)

# 迁移函数类型
MigrationFunc = Callable[[sqlite3.Connection], None]


def _migrate_v1(conn: sqlite3.Connection) -> None:
    """v1: 创建所有基础表"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_daily (
            symbol       TEXT NOT NULL,
            trade_date   TEXT NOT NULL,
            open         REAL NOT NULL,
            high         REAL NOT NULL,
            low          REAL NOT NULL,
            close        REAL NOT NULL,
            volume       REAL NOT NULL,
            amount       REAL DEFAULT 0,
            turnover     REAL DEFAULT 0,
            pct_change   REAL DEFAULT 0,
            pre_close    REAL DEFAULT 0,
            adj_factor   REAL DEFAULT 1.0,
            created_at   TEXT NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        );

        CREATE TABLE IF NOT EXISTS stock_info (
            symbol       TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            market       TEXT DEFAULT '',
            industry     TEXT DEFAULT '',
            sector       TEXT DEFAULT '',
            list_date    TEXT,
            delist_date  TEXT,
            is_active    INTEGER DEFAULT 1,
            updated_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_record (
            trade_id      TEXT PRIMARY KEY,
            date          TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            side          TEXT NOT NULL,
            quantity      INTEGER NOT NULL,
            price         REAL NOT NULL,
            commission    REAL DEFAULT 0,
            tax           REAL DEFAULT 0,
            pnl           REAL DEFAULT 0,
            strategy_name TEXT DEFAULT '',
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS backtest_result (
            result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name  TEXT NOT NULL,
            start_date     TEXT,
            end_date       TEXT,
            initial_capital REAL DEFAULT 0,
            final_value    REAL DEFAULT 0,
            total_return   REAL DEFAULT 0,
            annual_return  REAL DEFAULT 0,
            max_drawdown   REAL DEFAULT 0,
            sharpe_ratio   REAL DEFAULT 0,
            win_rate       REAL DEFAULT 0,
            total_trades   INTEGER DEFAULT 0,
            params_json    TEXT DEFAULT '{}',
            created_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_param (
            param_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name  TEXT NOT NULL,
            params_json    TEXT DEFAULT '{}',
            description    TEXT DEFAULT '',
            is_active      INTEGER DEFAULT 1,
            created_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshot (
            snapshot_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date  TEXT,
            cash           REAL DEFAULT 0,
            market_value   REAL DEFAULT 0,
            total_value    REAL DEFAULT 0,
            total_return   REAL DEFAULT 0,
            positions_json TEXT DEFAULT '{}',
            created_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            item_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol         TEXT NOT NULL UNIQUE,
            name           TEXT DEFAULT '',
            notes          TEXT DEFAULT '',
            target_price   REAL,
            stop_loss_price REAL,
            added_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alert (
            alert_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            alert_type    TEXT NOT NULL,
            threshold     REAL DEFAULT 0,
            message       TEXT DEFAULT '',
            is_triggered  INTEGER DEFAULT 0,
            is_active     INTEGER DEFAULT 1,
            created_at    TEXT NOT NULL,
            triggered_at  TEXT
        );
    """)
    logger.info("迁移 v1 完成: 创建所有基础表")


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """v2: 添加索引以优化查询性能"""
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_stock_daily_symbol
            ON stock_daily(symbol);
        CREATE INDEX IF NOT EXISTS idx_stock_daily_date
            ON stock_daily(trade_date);

        CREATE INDEX IF NOT EXISTS idx_trade_record_symbol
            ON trade_record(symbol);
        CREATE INDEX IF NOT EXISTS idx_trade_record_date
            ON trade_record(date);
        CREATE INDEX IF NOT EXISTS idx_trade_record_strategy
            ON trade_record(strategy_name);

        CREATE INDEX IF NOT EXISTS idx_backtest_result_strategy
            ON backtest_result(strategy_name);
        CREATE INDEX IF NOT EXISTS idx_backtest_result_return
            ON backtest_result(total_return);

        CREATE INDEX IF NOT EXISTS idx_portfolio_snapshot_date
            ON portfolio_snapshot(snapshot_date);

        CREATE INDEX IF NOT EXISTS idx_alert_symbol
            ON alert(symbol);
        CREATE INDEX IF NOT EXISTS idx_alert_active
            ON alert(is_active);
    """)
    logger.info("迁移 v2 完成: 添加索引")


# 迁移注册表: 版本号 -> 迁移函数
MIGRATIONS: dict[int, MigrationFunc] = {
    1: _migrate_v1,
    2: _migrate_v2,
}


class MigrationManager:
    """数据库迁移管理器"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ensure_version_table()

    def _ensure_version_table(self) -> None:
        """确保 schema_version 表存在"""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def get_current_version(self) -> int:
        """获取当前数据库版本"""
        cursor = self._conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        row = cursor.fetchone()
        return row[0] if row[0] is not None else 0

    def run_all(self) -> None:
        """执行所有未完成的迁移"""
        current = self.get_current_version()
        target = max(MIGRATIONS.keys()) if MIGRATIONS else 0

        if current >= target:
            logger.debug("数据库已是最新版本 v%d", current)
            return

        for version in range(current + 1, target + 1):
            if version not in MIGRATIONS:
                logger.warning("缺少迁移 v%d，跳过", version)
                continue
            self._run_migration(version)

    def _run_migration(self, version: int) -> None:
        """执行单个迁移"""
        logger.info("开始执行迁移 v%d ...", version)
        migrate_func = MIGRATIONS[version]
        migrate_func(self._conn)
        # 记录版本
        from datetime import datetime
        self._conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, datetime.now().isoformat()),
        )
        self._conn.commit()
        logger.info("迁移 v%d 已完成", version)

    def get_migration_history(self) -> list[tuple[int, str]]:
        """获取迁移历史"""
        cursor = self._conn.execute(
            "SELECT version, applied_at FROM schema_version ORDER BY version"
        )
        return cursor.fetchall()
