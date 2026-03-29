"""数据仓库层 - 封装所有 CRUD 操作

每个 Repository 接收 DatabaseManager 实例，通过参数化查询与数据库交互。
支持 pandas DataFrame 的输入输出。
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from quant_trading.database.models import (
    Alert,
    BacktestResult,
    PortfolioSnapshot,
    TradeRecord,
    WatchlistItem,
)

if TYPE_CHECKING:
    import pandas as pd

    from quant_trading.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


# ===========================================================================
# StockDataRepository
# ===========================================================================


class StockDataRepository:
    """股票行情数据仓库"""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # -- 写入 ---------------------------------------------------------------

    def save_daily_data(self, df: "pd.DataFrame") -> int:
        """将 DataFrame 批量写入 stock_daily 表

        DataFrame 必须包含列: symbol, trade_date, open, high, low, close, volume
        可选列: amount, turnover, pct_change, pre_close, adj_factor

        使用 INSERT OR REPLACE 实现 upsert。

        Returns:
            插入/更新的行数
        """
        required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame 缺少必需列: {missing}")

        # 标准化列
        optional_defaults = {
            "amount": 0.0,
            "turnover": 0.0,
            "pct_change": 0.0,
            "pre_close": 0.0,
            "adj_factor": 1.0,
        }
        for col, default in optional_defaults.items():
            if col not in df.columns:
                df = df.copy()
                df[col] = default

        now = datetime.now().isoformat()

        rows = []
        for _, row in df.iterrows():
            td = row["trade_date"]
            if hasattr(td, "isoformat"):
                td = td.isoformat()
            else:
                td = str(td)
            # 只保留日期部分 (去掉可能的时间部分)
            if len(td) > 10:
                td = td[:10]

            rows.append((
                str(row["symbol"]),
                td,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                float(row.get("amount", 0)),
                float(row.get("turnover", 0)),
                float(row.get("pct_change", 0)),
                float(row.get("pre_close", 0)),
                float(row.get("adj_factor", 1.0)),
                now,
            ))

        sql = """
            INSERT OR REPLACE INTO stock_daily
            (symbol, trade_date, open, high, low, close, volume,
             amount, turnover, pct_change, pre_close, adj_factor, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._db.executemany(sql, rows)
        self._db.commit()
        logger.info("保存 %d 条日线数据", len(rows))
        return len(rows)

    # -- 查询 ---------------------------------------------------------------

    def get_daily_data(
        self,
        symbol: str,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> "pd.DataFrame":
        """查询日线数据，返回 DataFrame

        Args:
            symbol: 股票代码
            start_date: 开始日期 (含)
            end_date: 结束日期 (含)

        Returns:
            包含行情数据的 DataFrame，按日期升序排列
        """
        import pandas as pd

        conditions = ["symbol = ?"]
        params: list[Any] = [symbol]

        if start_date:
            conditions.append("trade_date >= ?")
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date)
            )
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(
                end_date.isoformat()
                if hasattr(end_date, "isoformat")
                else str(end_date)
            )

        where = " AND ".join(conditions)
        sql = f"""
            SELECT symbol, trade_date, open, high, low, close, volume,
                   amount, turnover, pct_change, pre_close, adj_factor, created_at
            FROM stock_daily
            WHERE {where}
            ORDER BY trade_date ASC
        """
        rows = self._db.fetchall(sql, tuple(params))

        columns = [
            "symbol", "trade_date", "open", "high", "low", "close", "volume",
            "amount", "turnover", "pct_change", "pre_close", "adj_factor", "created_at",
        ]
        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def get_latest_date(self, symbol: str) -> str | None:
        """获取某股票最新数据日期 (用于增量更新)"""
        row = self._db.fetchone(
            "SELECT MAX(trade_date) FROM stock_daily WHERE symbol = ?",
            (symbol,),
        )
        return row[0] if row and row[0] else None

    def incremental_update(
        self, symbol: str, df: "pd.DataFrame"
    ) -> int:
        """增量更新: 只插入比现有数据更新的记录

        Args:
            symbol: 股票代码
            df: 新数据的 DataFrame

        Returns:
            新插入的行数
        """
        import pandas as pd

        latest = self.get_latest_date(symbol)
        if latest and not df.empty:
            df = df.copy()
            # 确保 trade_date 列可比较
            if hasattr(df["trade_date"].iloc[0], "isoformat"):
                mask = df["trade_date"].apply(
                    lambda x: str(x.isoformat())[:10] if hasattr(x, "isoformat") else str(x)[:10]
                ) > latest
            else:
                mask = df["trade_date"].astype(str).str[:10] > latest
            df = df[mask]

        if df.empty:
            logger.info("股票 %s 无需更新", symbol)
            return 0

        return self.save_daily_data(df)

    def get_symbols(self) -> list[str]:
        """获取所有有数据的股票代码"""
        rows = self._db.fetchall(
            "SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol"
        )
        return [r[0] for r in rows]

    def delete_daily_data(self, symbol: str) -> int:
        """删除指定股票的所有日线数据"""
        cursor = self._db.execute(
            "DELETE FROM stock_daily WHERE symbol = ?", (symbol,)
        )
        self._db.commit()
        return cursor.rowcount


# ===========================================================================
# TradeRepository
# ===========================================================================


class TradeRepository:
    """交易记录仓库"""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def save_trade(self, trade: TradeRecord) -> None:
        """保存单条交易记录"""
        sql = """
            INSERT OR REPLACE INTO trade_record
            (trade_id, date, symbol, side, quantity, price,
             commission, tax, pnl, strategy_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._db.execute(sql, (
            trade.trade_id,
            trade.date.isoformat(),
            trade.symbol,
            trade.side,
            trade.quantity,
            trade.price,
            trade.commission,
            trade.tax,
            trade.pnl,
            trade.strategy_name,
            trade.created_at.isoformat(),
        ))
        self._db.commit()

    def save_trades(self, trades: list[TradeRecord]) -> int:
        """批量保存交易记录"""
        sql = """
            INSERT OR REPLACE INTO trade_record
            (trade_id, date, symbol, side, quantity, price,
             commission, tax, pnl, strategy_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                t.trade_id, t.date.isoformat(), t.symbol, t.side,
                t.quantity, t.price, t.commission, t.tax, t.pnl,
                t.strategy_name, t.created_at.isoformat(),
            )
            for t in trades
        ]
        self._db.executemany(sql, params)
        self._db.commit()
        return len(params)

    def get_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        side: str | None = None,
    ) -> list[TradeRecord]:
        """查询交易记录 (支持多条件过滤)"""
        conditions: list[str] = []
        params: list[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if strategy:
            conditions.append("strategy_name = ?")
            params.append(strategy)
        if start_date:
            conditions.append("date >= ?")
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date)
            )
        if end_date:
            conditions.append("date <= ?")
            params.append(
                end_date.isoformat()
                if hasattr(end_date, "isoformat")
                else str(end_date)
            )
        if side:
            conditions.append("side = ?")
            params.append(side)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT trade_id, date, symbol, side, quantity, price,
                   commission, tax, pnl, strategy_name, created_at
            FROM trade_record
            {where}
            ORDER BY date ASC, created_at ASC
        """
        rows = self._db.fetchall(sql, tuple(params) if params else None)
        return [TradeRecord.from_row(r) for r in rows]

    def get_trade_summary(
        self, strategy: str | None = None
    ) -> dict[str, Any]:
        """获取交易汇总统计

        Returns:
            包含: total_trades, buy_count, sell_count, total_pnl,
                  win_count, loss_count, win_rate, avg_pnl
        """
        condition = "WHERE strategy_name = ?" if strategy else ""
        params = (strategy,) if strategy else None

        total = self._db.fetchval(
            f"SELECT COUNT(*) FROM trade_record {condition}", params
        ) or 0
        buy_count = self._db.fetchval(
            f"SELECT COUNT(*) FROM trade_record {condition}"
            + (" AND" if condition else "WHERE") + " side = 'buy'",
            params,
        ) or 0
        sell_count = self._db.fetchval(
            f"SELECT COUNT(*) FROM trade_record {condition}"
            + (" AND" if condition else "WHERE") + " side = 'sell'",
            params,
        ) or 0
        total_pnl = self._db.fetchval(
            f"SELECT COALESCE(SUM(pnl), 0) FROM trade_record {condition}",
            params,
        ) or 0.0
        win_count = self._db.fetchval(
            f"SELECT COUNT(*) FROM trade_record {condition}"
            + (" AND" if condition else "WHERE") + " pnl > 0",
            params,
        ) or 0
        loss_count = self._db.fetchval(
            f"SELECT COUNT(*) FROM trade_record {condition}"
            + (" AND" if condition else "WHERE") + " pnl < 0",
            params,
        ) or 0

        # 胜率只算有盈亏的 (sell 单)
        pnl_trades = win_count + loss_count
        win_rate = (win_count / pnl_trades * 100) if pnl_trades > 0 else 0.0
        avg_pnl = (total_pnl / pnl_trades) if pnl_trades > 0 else 0.0

        return {
            "total_trades": total,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_pnl": total_pnl,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
        }

    def export_trades(
        self,
        strategy: str | None = None,
        format: str = "csv",
    ) -> str:
        """导出交易记录

        Args:
            strategy: 可选策略名过滤
            format: "csv" 或 "json"

        Returns:
            CSV 或 JSON 字符串
        """
        trades = self.get_trades(strategy=strategy)

        if format == "json":
            return json.dumps(
                [t.to_dict() for t in trades],
                ensure_ascii=False,
                indent=2,
            )

        # CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "trade_id", "date", "symbol", "side", "quantity", "price",
            "commission", "tax", "pnl", "strategy_name", "created_at",
        ])
        for t in trades:
            writer.writerow([
                t.trade_id, t.date.isoformat(), t.symbol, t.side,
                t.quantity, t.price, t.commission, t.tax, t.pnl,
                t.strategy_name, t.created_at.isoformat(),
            ])
        return output.getvalue()

    def delete_trade(self, trade_id: str) -> bool:
        """删除单条交易记录"""
        cursor = self._db.execute(
            "DELETE FROM trade_record WHERE trade_id = ?", (trade_id,)
        )
        self._db.commit()
        return cursor.rowcount > 0


# ===========================================================================
# BacktestRepository
# ===========================================================================


class BacktestRepository:
    """回测结果仓库"""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def save_result(self, result: BacktestResult) -> int:
        """保存回测结果，返回 result_id"""
        sql = """
            INSERT INTO backtest_result
            (strategy_name, start_date, end_date, initial_capital, final_value,
             total_return, annual_return, max_drawdown, sharpe_ratio,
             win_rate, total_trades, params_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor = self._db.execute(sql, (
            result.strategy_name,
            result.start_date.isoformat() if result.start_date else None,
            result.end_date.isoformat() if result.end_date else None,
            result.initial_capital,
            result.final_value,
            result.total_return,
            result.annual_return,
            result.max_drawdown,
            result.sharpe_ratio,
            result.win_rate,
            result.total_trades,
            result.params_json,
            result.created_at.isoformat(),
        ))
        self._db.commit()
        result.result_id = cursor.lastrowid
        return cursor.lastrowid or 0

    def get_results(
        self,
        strategy: str | None = None,
        limit: int = 50,
    ) -> list[BacktestResult]:
        """查询回测结果"""
        if strategy:
            sql = """
                SELECT result_id, strategy_name, start_date, end_date,
                       initial_capital, final_value, total_return, annual_return,
                       max_drawdown, sharpe_ratio, win_rate, total_trades,
                       params_json, created_at
                FROM backtest_result
                WHERE strategy_name = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            rows = self._db.fetchall(sql, (strategy, limit))
        else:
            sql = """
                SELECT result_id, strategy_name, start_date, end_date,
                       initial_capital, final_value, total_return, annual_return,
                       max_drawdown, sharpe_ratio, win_rate, total_trades,
                       params_json, created_at
                FROM backtest_result
                ORDER BY created_at DESC
                LIMIT ?
            """
            rows = self._db.fetchall(sql, (limit,))

        return [BacktestResult.from_row(r) for r in rows]

    def get_best_result(
        self,
        strategy: str | None = None,
        metric: str = "sharpe_ratio",
    ) -> BacktestResult | None:
        """获取最优回测结果

        Args:
            strategy: 可选策略名过滤
            metric: 排序指标 (sharpe_ratio, total_return, max_drawdown, win_rate)
        """
        allowed_metrics = {
            "sharpe_ratio", "total_return", "annual_return",
            "max_drawdown", "win_rate",
        }
        if metric not in allowed_metrics:
            raise ValueError(f"不支持的指标: {metric}, 可选: {allowed_metrics}")

        # max_drawdown 越小越好，其他越大越好
        order = "ASC" if metric == "max_drawdown" else "DESC"

        if strategy:
            sql = f"""
                SELECT result_id, strategy_name, start_date, end_date,
                       initial_capital, final_value, total_return, annual_return,
                       max_drawdown, sharpe_ratio, win_rate, total_trades,
                       params_json, created_at
                FROM backtest_result
                WHERE strategy_name = ?
                ORDER BY {metric} {order}
                LIMIT 1
            """
            row = self._db.fetchone(sql, (strategy,))
        else:
            sql = f"""
                SELECT result_id, strategy_name, start_date, end_date,
                       initial_capital, final_value, total_return, annual_return,
                       max_drawdown, sharpe_ratio, win_rate, total_trades,
                       params_json, created_at
                FROM backtest_result
                ORDER BY {metric} {order}
                LIMIT 1
            """
            row = self._db.fetchone(sql)

        return BacktestResult.from_row(row) if row else None

    def compare_results(
        self, result_ids: list[int]
    ) -> list[BacktestResult]:
        """比较多个回测结果"""
        if not result_ids:
            return []
        placeholders = ",".join(["?"] * len(result_ids))
        sql = f"""
            SELECT result_id, strategy_name, start_date, end_date,
                   initial_capital, final_value, total_return, annual_return,
                   max_drawdown, sharpe_ratio, win_rate, total_trades,
                   params_json, created_at
            FROM backtest_result
            WHERE result_id IN ({placeholders})
            ORDER BY total_return DESC
        """
        rows = self._db.fetchall(sql, tuple(result_ids))
        return [BacktestResult.from_row(r) for r in rows]

    def delete_result(self, result_id: int) -> bool:
        """删除回测结果"""
        cursor = self._db.execute(
            "DELETE FROM backtest_result WHERE result_id = ?", (result_id,)
        )
        self._db.commit()
        return cursor.rowcount > 0


# ===========================================================================
# PortfolioRepository
# ===========================================================================


class PortfolioRepository:
    """投资组合快照仓库"""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> int:
        """保存组合快照，返回 snapshot_id"""
        sql = """
            INSERT INTO portfolio_snapshot
            (snapshot_date, cash, market_value, total_value,
             total_return, positions_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor = self._db.execute(sql, (
            snapshot.snapshot_date.isoformat() if snapshot.snapshot_date else None,
            snapshot.cash,
            snapshot.market_value,
            snapshot.total_value,
            snapshot.total_return,
            snapshot.positions_json,
            snapshot.created_at.isoformat(),
        ))
        self._db.commit()
        snapshot.snapshot_id = cursor.lastrowid
        return cursor.lastrowid or 0

    def get_snapshots(
        self,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        limit: int = 100,
    ) -> list[PortfolioSnapshot]:
        """查询组合快照"""
        conditions: list[str] = []
        params: list[Any] = []

        if start_date:
            conditions.append("snapshot_date >= ?")
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date)
            )
        if end_date:
            conditions.append("snapshot_date <= ?")
            params.append(
                end_date.isoformat()
                if hasattr(end_date, "isoformat")
                else str(end_date)
            )

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT snapshot_id, snapshot_date, cash, market_value,
                   total_value, total_return, positions_json, created_at
            FROM portfolio_snapshot
            {where}
            ORDER BY snapshot_date DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self._db.fetchall(sql, tuple(params))
        return [PortfolioSnapshot.from_row(r) for r in rows]

    def get_latest(self) -> PortfolioSnapshot | None:
        """获取最新快照"""
        row = self._db.fetchone("""
            SELECT snapshot_id, snapshot_date, cash, market_value,
                   total_value, total_return, positions_json, created_at
            FROM portfolio_snapshot
            ORDER BY snapshot_date DESC, snapshot_id DESC
            LIMIT 1
        """)
        return PortfolioSnapshot.from_row(row) if row else None

    def get_history(
        self, days: int = 30
    ) -> list[PortfolioSnapshot]:
        """获取最近 N 天的快照历史"""
        sql = """
            SELECT snapshot_id, snapshot_date, cash, market_value,
                   total_value, total_return, positions_json, created_at
            FROM portfolio_snapshot
            ORDER BY snapshot_date DESC, snapshot_id DESC
            LIMIT ?
        """
        rows = self._db.fetchall(sql, (days,))
        # 返回时按日期升序
        return [PortfolioSnapshot.from_row(r) for r in reversed(rows)]

    def delete_snapshot(self, snapshot_id: int) -> bool:
        """删除快照"""
        cursor = self._db.execute(
            "DELETE FROM portfolio_snapshot WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        self._db.commit()
        return cursor.rowcount > 0


# ===========================================================================
# WatchlistRepository
# ===========================================================================


class WatchlistRepository:
    """自选股仓库"""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def add(self, item: WatchlistItem) -> int:
        """添加自选股，返回 item_id"""
        sql = """
            INSERT OR REPLACE INTO watchlist
            (symbol, name, notes, target_price, stop_loss_price, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor = self._db.execute(sql, (
            item.symbol,
            item.name,
            item.notes,
            item.target_price,
            item.stop_loss_price,
            item.added_at.isoformat(),
        ))
        self._db.commit()
        item.item_id = cursor.lastrowid
        return cursor.lastrowid or 0

    def remove(self, symbol: str) -> bool:
        """移除自选股 (按 symbol)"""
        cursor = self._db.execute(
            "DELETE FROM watchlist WHERE symbol = ?", (symbol,)
        )
        self._db.commit()
        return cursor.rowcount > 0

    def get(self, symbol: str) -> WatchlistItem | None:
        """获取单只自选股"""
        row = self._db.fetchone(
            """
            SELECT item_id, symbol, name, notes,
                   target_price, stop_loss_price, added_at
            FROM watchlist
            WHERE symbol = ?
            """,
            (symbol,),
        )
        return WatchlistItem.from_row(row) if row else None

    def get_all(self) -> list[WatchlistItem]:
        """获取所有自选股"""
        rows = self._db.fetchall("""
            SELECT item_id, symbol, name, notes,
                   target_price, stop_loss_price, added_at
            FROM watchlist
            ORDER BY added_at DESC
        """)
        return [WatchlistItem.from_row(r) for r in rows]

    def update(
        self,
        symbol: str,
        name: str | None = None,
        notes: str | None = None,
        target_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> bool:
        """更新自选股信息"""
        updates: list[str] = []
        params: list[Any] = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if target_price is not None:
            updates.append("target_price = ?")
            params.append(target_price)
        if stop_loss_price is not None:
            updates.append("stop_loss_price = ?")
            params.append(stop_loss_price)

        if not updates:
            return False

        params.append(symbol)
        sql = f"UPDATE watchlist SET {', '.join(updates)} WHERE symbol = ?"
        cursor = self._db.execute(sql, tuple(params))
        self._db.commit()
        return cursor.rowcount > 0

    def contains(self, symbol: str) -> bool:
        """检查是否在自选股中"""
        val = self._db.fetchval(
            "SELECT COUNT(*) FROM watchlist WHERE symbol = ?", (symbol,)
        )
        return bool(val and val > 0)

    def count(self) -> int:
        """自选股数量"""
        return self._db.fetchval("SELECT COUNT(*) FROM watchlist") or 0


# ===========================================================================
# AlertRepository (补充)
# ===========================================================================


class AlertRepository:
    """告警仓库"""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def add(self, alert: Alert) -> int:
        """添加告警，返回 alert_id"""
        sql = """
            INSERT INTO alert
            (symbol, alert_type, threshold, message,
             is_triggered, is_active, created_at, triggered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor = self._db.execute(sql, (
            alert.symbol,
            alert.alert_type,
            alert.threshold,
            alert.message,
            int(alert.is_triggered),
            int(alert.is_active),
            alert.created_at.isoformat(),
            alert.triggered_at.isoformat() if alert.triggered_at else None,
        ))
        self._db.commit()
        alert.alert_id = cursor.lastrowid
        return cursor.lastrowid or 0

    def get_active(self, symbol: str | None = None) -> list[Alert]:
        """获取活跃告警"""
        if symbol:
            sql = """
                SELECT alert_id, symbol, alert_type, threshold, message,
                       is_triggered, is_active, created_at, triggered_at
                FROM alert
                WHERE is_active = 1 AND symbol = ?
                ORDER BY created_at DESC
            """
            rows = self._db.fetchall(sql, (symbol,))
        else:
            sql = """
                SELECT alert_id, symbol, alert_type, threshold, message,
                       is_triggered, is_active, created_at, triggered_at
                FROM alert
                WHERE is_active = 1
                ORDER BY created_at DESC
            """
            rows = self._db.fetchall(sql)

        return [Alert.from_row(r) for r in rows]

    def trigger(self, alert_id: int) -> bool:
        """触发告警"""
        now = datetime.now().isoformat()
        cursor = self._db.execute(
            "UPDATE alert SET is_triggered = 1, triggered_at = ? WHERE alert_id = ?",
            (now, alert_id),
        )
        self._db.commit()
        return cursor.rowcount > 0

    def deactivate(self, alert_id: int) -> bool:
        """停用告警"""
        cursor = self._db.execute(
            "UPDATE alert SET is_active = 0 WHERE alert_id = ?",
            (alert_id,),
        )
        self._db.commit()
        return cursor.rowcount > 0

    def delete(self, alert_id: int) -> bool:
        """删除告警"""
        cursor = self._db.execute(
            "DELETE FROM alert WHERE alert_id = ?", (alert_id,)
        )
        self._db.commit()
        return cursor.rowcount > 0
