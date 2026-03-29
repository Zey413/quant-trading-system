"""数据库模块完整测试

使用 :memory: SQLite，测试所有 CRUD、DataFrame 转换、迁移。
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest

from quant_trading.database.connection import DatabaseManager
from quant_trading.database.migrations import MigrationManager
from quant_trading.database.models import (
    Alert,
    BacktestResult,
    PortfolioSnapshot,
    StockDaily,
    StockInfo,
    TradeRecord,
    WatchlistItem,
)
from quant_trading.database.repositories import (
    AlertRepository,
    BacktestRepository,
    PortfolioRepository,
    StockDataRepository,
    TradeRepository,
    WatchlistRepository,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def db():
    """创建内存数据库"""
    with DatabaseManager(":memory:") as manager:
        yield manager


@pytest.fixture
def stock_repo(db):
    return StockDataRepository(db)


@pytest.fixture
def trade_repo(db):
    return TradeRepository(db)


@pytest.fixture
def backtest_repo(db):
    return BacktestRepository(db)


@pytest.fixture
def portfolio_repo(db):
    return PortfolioRepository(db)


@pytest.fixture
def watchlist_repo(db):
    return WatchlistRepository(db)


@pytest.fixture
def alert_repo(db):
    return AlertRepository(db)


@pytest.fixture
def sample_df():
    """创建示例行情 DataFrame"""
    return pd.DataFrame({
        "symbol": ["000001"] * 5,
        "trade_date": pd.date_range("2024-01-01", periods=5),
        "open": [10.0, 10.2, 10.5, 10.3, 10.6],
        "high": [10.5, 10.8, 11.0, 10.7, 11.1],
        "low": [9.8, 10.0, 10.3, 10.1, 10.4],
        "close": [10.2, 10.5, 10.3, 10.6, 10.8],
        "volume": [1e6, 1.2e6, 0.8e6, 1.1e6, 1.5e6],
        "amount": [1e7, 1.2e7, 0.8e7, 1.1e7, 1.5e7],
    })


# ===========================================================================
# 1. DatabaseManager 基础测试
# ===========================================================================


class TestDatabaseManager:
    """数据库管理器测试"""

    def test_connect_and_close(self):
        """测试连接和关闭"""
        db = DatabaseManager(":memory:")
        db.connect()
        assert db.connection is not None
        db.close()

    def test_context_manager(self):
        """测试 context manager"""
        with DatabaseManager(":memory:") as db:
            result = db.fetchval("SELECT 1")
            assert result == 1

    def test_wal_mode(self, db):
        """测试 WAL 模式 (内存数据库回退为 memory)"""
        mode = db.fetchval("PRAGMA journal_mode")
        # :memory: 数据库不支持 WAL，会回退为 memory
        assert mode in ("wal", "memory")

    def test_tables_created(self, db):
        """测试自动建表"""
        tables = db.get_table_names()
        expected = {
            "stock_daily", "stock_info", "trade_record",
            "backtest_result", "strategy_param", "portfolio_snapshot",
            "watchlist", "alert", "schema_version",
        }
        assert expected.issubset(set(tables))

    def test_table_exists(self, db):
        """测试表存在性检查"""
        assert db.table_exists("stock_daily") is True
        assert db.table_exists("nonexistent_table") is False

    def test_transaction_commit(self, db):
        """测试事务提交"""
        with db.transaction():
            db.execute(
                "INSERT INTO watchlist (symbol, name, added_at) VALUES (?, ?, ?)",
                ("000001", "平安银行", datetime.now().isoformat()),
            )
        count = db.fetchval("SELECT COUNT(*) FROM watchlist")
        assert count == 1

    def test_transaction_rollback(self, db):
        """测试事务回滚"""
        try:
            with db.transaction():
                db.execute(
                    "INSERT INTO watchlist (symbol, name, added_at) VALUES (?, ?, ?)",
                    ("000001", "平安银行", datetime.now().isoformat()),
                )
                raise ValueError("模拟错误")
        except ValueError:
            pass
        count = db.fetchval("SELECT COUNT(*) FROM watchlist")
        assert count == 0

    def test_get_row_count(self, db):
        """测试行数统计"""
        count = db.get_row_count("watchlist")
        assert count == 0


# ===========================================================================
# 2. Models 测试
# ===========================================================================


class TestModels:
    """数据模型测试"""

    def test_stock_daily_to_dict(self):
        """测试 StockDaily 序列化"""
        sd = StockDaily(
            symbol="000001",
            trade_date=date(2024, 1, 1),
            open=10.0, high=10.5, low=9.8,
            close=10.2, volume=1e6,
        )
        d = sd.to_dict()
        assert d["symbol"] == "000001"
        assert d["trade_date"] == "2024-01-01"
        assert d["close"] == 10.2

    def test_stock_daily_from_row(self):
        """测试 StockDaily 从行构造"""
        now = datetime.now()
        row = (
            "000001", "2024-01-01", 10.0, 10.5, 9.8,
            10.2, 1e6, 1e7, 0.5, 2.0,
            10.0, 1.0, now.isoformat(),
        )
        sd = StockDaily.from_row(row)
        assert sd.symbol == "000001"
        assert sd.trade_date == date(2024, 1, 1)
        assert sd.close == 10.2

    def test_trade_record_total_cost(self):
        """测试成交记录总成本"""
        tr = TradeRecord(
            trade_id="T001",
            date=date(2024, 1, 1),
            symbol="000001",
            side="buy",
            quantity=1000,
            price=10.0,
            commission=5.0,
            tax=0.0,
        )
        assert tr.total_cost == 10005.0

    def test_trade_record_from_core_model(self):
        """测试从 core.models.TradeRecord 转换"""
        from quant_trading.core.models import TradeRecord as CoreTradeRecord

        core = CoreTradeRecord(
            trade_id="T001",
            date=date(2024, 1, 1),
            symbol="000001",
            side="buy",
            quantity=1000,
            price=10.0,
            commission=5.0,
            tax=0.0,
            pnl=500.0,
            strategy_name="ma_cross",
        )
        db_record = TradeRecord.from_core_model(core)
        assert db_record.trade_id == "T001"
        assert db_record.pnl == 500.0
        assert db_record.strategy_name == "ma_cross"

    def test_backtest_result_to_dict(self):
        """测试回测结果序列化"""
        br = BacktestResult(
            strategy_name="test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            total_return=15.5,
        )
        d = br.to_dict()
        assert d["strategy_name"] == "test"
        assert d["start_date"] == "2024-01-01"

    def test_alert_to_dict(self):
        """测试告警序列化"""
        a = Alert(
            symbol="000001",
            alert_type="price_above",
            threshold=15.0,
            message="价格突破15元",
        )
        d = a.to_dict()
        assert d["is_triggered"] == 0
        assert d["is_active"] == 1


# ===========================================================================
# 3. StockDataRepository 测试
# ===========================================================================


class TestStockDataRepository:
    """股票数据仓库测试"""

    def test_save_daily_data(self, stock_repo, sample_df):
        """测试批量保存日线数据"""
        count = stock_repo.save_daily_data(sample_df)
        assert count == 5

    def test_get_daily_data(self, stock_repo, sample_df):
        """测试查询日线数据"""
        stock_repo.save_daily_data(sample_df)
        df = stock_repo.get_daily_data("000001")
        assert len(df) == 5
        assert "close" in df.columns
        assert df.iloc[0]["close"] == 10.2

    def test_get_daily_data_with_date_range(self, stock_repo, sample_df):
        """测试按日期范围查询"""
        stock_repo.save_daily_data(sample_df)
        df = stock_repo.get_daily_data(
            "000001",
            start_date="2024-01-02",
            end_date="2024-01-04",
        )
        assert len(df) == 3

    def test_get_daily_data_empty(self, stock_repo):
        """测试查询空结果"""
        df = stock_repo.get_daily_data("999999")
        assert df.empty

    def test_incremental_update(self, stock_repo, sample_df):
        """测试增量更新"""
        stock_repo.save_daily_data(sample_df)
        # 原数据最新日期是 2024-01-05

        # 创建新数据: 2024-01-04, 01-05, 01-06
        # 只有 01-06 严格大于最新日期 01-05
        new_df = pd.DataFrame({
            "symbol": ["000001"] * 3,
            "trade_date": pd.date_range("2024-01-04", periods=3),
            "open": [10.7, 10.8, 10.9],
            "high": [11.0, 11.1, 11.2],
            "low": [10.5, 10.6, 10.7],
            "close": [10.9, 11.0, 11.1],
            "volume": [1e6, 1e6, 1e6],
        })
        count = stock_repo.incremental_update("000001", new_df)
        # 只有 2024-01-06 是新数据 (> 2024-01-05)
        assert count == 1

        # 验证总数据量
        df = stock_repo.get_daily_data("000001")
        assert len(df) == 6  # 原5条 + 新1条

    def test_get_symbols(self, stock_repo, sample_df):
        """测试获取所有股票代码"""
        stock_repo.save_daily_data(sample_df)
        # 再加一只股票
        df2 = sample_df.copy()
        df2["symbol"] = "600000"
        stock_repo.save_daily_data(df2)
        symbols = stock_repo.get_symbols()
        assert "000001" in symbols
        assert "600000" in symbols
        assert len(symbols) == 2

    def test_delete_daily_data(self, stock_repo, sample_df):
        """测试删除日线数据"""
        stock_repo.save_daily_data(sample_df)
        deleted = stock_repo.delete_daily_data("000001")
        assert deleted == 5
        df = stock_repo.get_daily_data("000001")
        assert df.empty

    def test_save_missing_required_columns(self, stock_repo):
        """测试缺少必需列时报错"""
        bad_df = pd.DataFrame({"symbol": ["000001"], "close": [10.0]})
        with pytest.raises(ValueError, match="缺少必需列"):
            stock_repo.save_daily_data(bad_df)

    def test_get_latest_date(self, stock_repo, sample_df):
        """测试获取最新日期"""
        stock_repo.save_daily_data(sample_df)
        latest = stock_repo.get_latest_date("000001")
        assert latest == "2024-01-05"

    def test_upsert_behavior(self, stock_repo, sample_df):
        """测试 upsert: 重复 insert 应更新"""
        stock_repo.save_daily_data(sample_df)
        # 修改 close 价格
        updated_df = sample_df.copy()
        updated_df["close"] = 99.0
        stock_repo.save_daily_data(updated_df)
        df = stock_repo.get_daily_data("000001")
        assert len(df) == 5
        assert df.iloc[0]["close"] == 99.0


# ===========================================================================
# 4. TradeRepository 测试
# ===========================================================================


class TestTradeRepository:
    """交易记录仓库测试"""

    def _make_trade(self, trade_id="T001", symbol="000001",
                    side="buy", pnl=0.0) -> TradeRecord:
        return TradeRecord(
            trade_id=trade_id,
            date=date(2024, 1, 15),
            symbol=symbol,
            side=side,
            quantity=1000,
            price=10.0,
            commission=5.0,
            tax=0.0,
            pnl=pnl,
            strategy_name="ma_cross",
        )

    def test_save_and_get_trade(self, trade_repo):
        """测试保存和查询单条交易"""
        trade = self._make_trade()
        trade_repo.save_trade(trade)
        trades = trade_repo.get_trades(symbol="000001")
        assert len(trades) == 1
        assert trades[0].trade_id == "T001"
        assert trades[0].price == 10.0

    def test_save_trades_batch(self, trade_repo):
        """测试批量保存"""
        trades = [
            self._make_trade("T001", pnl=100),
            self._make_trade("T002", side="sell", pnl=-50),
            self._make_trade("T003", pnl=200),
        ]
        count = trade_repo.save_trades(trades)
        assert count == 3
        all_trades = trade_repo.get_trades()
        assert len(all_trades) == 3

    def test_get_trades_filter_by_side(self, trade_repo):
        """测试按买卖方向过滤"""
        trade_repo.save_trade(self._make_trade("T001", side="buy"))
        trade_repo.save_trade(self._make_trade("T002", side="sell"))
        buys = trade_repo.get_trades(side="buy")
        assert len(buys) == 1
        assert buys[0].side == "buy"

    def test_get_trade_summary(self, trade_repo):
        """测试交易汇总"""
        trades = [
            self._make_trade("T001", side="buy", pnl=0),
            self._make_trade("T002", side="sell", pnl=500),
            self._make_trade("T003", side="sell", pnl=-200),
            self._make_trade("T004", side="sell", pnl=300),
        ]
        trade_repo.save_trades(trades)
        summary = trade_repo.get_trade_summary()
        assert summary["total_trades"] == 4
        assert summary["buy_count"] == 1
        assert summary["sell_count"] == 3
        assert summary["total_pnl"] == 600.0
        assert summary["win_count"] == 2
        assert summary["loss_count"] == 1
        assert summary["win_rate"] == pytest.approx(66.6667, rel=0.01)

    def test_export_trades_csv(self, trade_repo):
        """测试导出 CSV"""
        trade_repo.save_trade(self._make_trade())
        csv_str = trade_repo.export_trades(format="csv")
        assert "trade_id" in csv_str
        assert "T001" in csv_str

    def test_export_trades_json(self, trade_repo):
        """测试导出 JSON"""
        trade_repo.save_trade(self._make_trade())
        json_str = trade_repo.export_trades(format="json")
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["trade_id"] == "T001"

    def test_delete_trade(self, trade_repo):
        """测试删除交易"""
        trade_repo.save_trade(self._make_trade())
        assert trade_repo.delete_trade("T001") is True
        assert trade_repo.delete_trade("T001") is False


# ===========================================================================
# 5. BacktestRepository 测试
# ===========================================================================


class TestBacktestRepository:
    """回测结果仓库测试"""

    def _make_result(self, name="ma_cross", total_return=10.0,
                     sharpe=1.5) -> BacktestResult:
        return BacktestResult(
            strategy_name=name,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=1_000_000,
            final_value=1_100_000,
            total_return=total_return,
            annual_return=total_return,
            max_drawdown=5.0,
            sharpe_ratio=sharpe,
            win_rate=60.0,
            total_trades=100,
            params_json='{"fast": 5, "slow": 20}',
        )

    def test_save_and_get_result(self, backtest_repo):
        """测试保存和查询"""
        result = self._make_result()
        rid = backtest_repo.save_result(result)
        assert rid > 0
        results = backtest_repo.get_results(strategy="ma_cross")
        assert len(results) == 1
        assert results[0].total_return == 10.0

    def test_get_best_result(self, backtest_repo):
        """测试获取最优结果"""
        backtest_repo.save_result(self._make_result(sharpe=1.0))
        backtest_repo.save_result(self._make_result(sharpe=2.5))
        backtest_repo.save_result(self._make_result(sharpe=1.8))
        best = backtest_repo.get_best_result(metric="sharpe_ratio")
        assert best is not None
        assert best.sharpe_ratio == 2.5

    def test_compare_results(self, backtest_repo):
        """测试比较回测结果"""
        id1 = backtest_repo.save_result(self._make_result(total_return=10))
        id2 = backtest_repo.save_result(self._make_result(total_return=20))
        id3 = backtest_repo.save_result(self._make_result(total_return=5))
        compared = backtest_repo.compare_results([id1, id2, id3])
        assert len(compared) == 3
        # 按 total_return DESC
        assert compared[0].total_return == 20
        assert compared[-1].total_return == 5

    def test_invalid_metric(self, backtest_repo):
        """测试不合法的指标"""
        with pytest.raises(ValueError, match="不支持的指标"):
            backtest_repo.get_best_result(metric="invalid")

    def test_delete_result(self, backtest_repo):
        """测试删除"""
        rid = backtest_repo.save_result(self._make_result())
        assert backtest_repo.delete_result(rid) is True
        results = backtest_repo.get_results()
        assert len(results) == 0


# ===========================================================================
# 6. PortfolioRepository 测试
# ===========================================================================


class TestPortfolioRepository:
    """组合快照仓库测试"""

    def _make_snapshot(self, d: date = date(2024, 6, 1),
                       total_value: float = 1_050_000) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            snapshot_date=d,
            cash=500_000,
            market_value=total_value - 500_000,
            total_value=total_value,
            total_return=5.0,
            positions_json='{"000001": {"qty": 1000, "price": 10.5}}',
        )

    def test_save_and_get_snapshot(self, portfolio_repo):
        """测试保存和查询快照"""
        sid = portfolio_repo.save_snapshot(self._make_snapshot())
        assert sid > 0
        snapshots = portfolio_repo.get_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].total_value == 1_050_000

    def test_get_latest(self, portfolio_repo):
        """测试获取最新快照"""
        portfolio_repo.save_snapshot(
            self._make_snapshot(date(2024, 1, 1), 1_000_000)
        )
        portfolio_repo.save_snapshot(
            self._make_snapshot(date(2024, 6, 1), 1_050_000)
        )
        portfolio_repo.save_snapshot(
            self._make_snapshot(date(2024, 12, 1), 1_100_000)
        )
        latest = portfolio_repo.get_latest()
        assert latest is not None
        assert latest.total_value == 1_100_000

    def test_get_history(self, portfolio_repo):
        """测试获取历史"""
        for i in range(10):
            portfolio_repo.save_snapshot(
                self._make_snapshot(date(2024, 1, i + 1), 1_000_000 + i * 10000)
            )
        history = portfolio_repo.get_history(days=5)
        assert len(history) == 5
        # 按日期升序
        assert history[0].total_value < history[-1].total_value

    def test_delete_snapshot(self, portfolio_repo):
        """测试删除快照"""
        sid = portfolio_repo.save_snapshot(self._make_snapshot())
        assert portfolio_repo.delete_snapshot(sid) is True
        assert portfolio_repo.get_latest() is None


# ===========================================================================
# 7. WatchlistRepository 测试
# ===========================================================================


class TestWatchlistRepository:
    """自选股仓库测试"""

    def test_add_and_get(self, watchlist_repo):
        """测试添加和查询"""
        item = WatchlistItem(symbol="000001", name="平安银行")
        iid = watchlist_repo.add(item)
        assert iid > 0
        got = watchlist_repo.get("000001")
        assert got is not None
        assert got.name == "平安银行"

    def test_remove(self, watchlist_repo):
        """测试移除"""
        watchlist_repo.add(WatchlistItem(symbol="000001", name="平安银行"))
        assert watchlist_repo.remove("000001") is True
        assert watchlist_repo.get("000001") is None

    def test_get_all(self, watchlist_repo):
        """测试获取全部"""
        watchlist_repo.add(WatchlistItem(symbol="000001", name="平安银行"))
        watchlist_repo.add(WatchlistItem(symbol="600000", name="浦发银行"))
        all_items = watchlist_repo.get_all()
        assert len(all_items) == 2

    def test_update(self, watchlist_repo):
        """测试更新"""
        watchlist_repo.add(WatchlistItem(symbol="000001", name="平安银行"))
        watchlist_repo.update("000001", notes="看好", target_price=15.0)
        item = watchlist_repo.get("000001")
        assert item is not None
        assert item.notes == "看好"
        assert item.target_price == 15.0

    def test_contains_and_count(self, watchlist_repo):
        """测试包含检查和计数"""
        assert watchlist_repo.contains("000001") is False
        assert watchlist_repo.count() == 0
        watchlist_repo.add(WatchlistItem(symbol="000001", name="平安银行"))
        assert watchlist_repo.contains("000001") is True
        assert watchlist_repo.count() == 1


# ===========================================================================
# 8. Migration 测试
# ===========================================================================


class TestMigrations:
    """迁移系统测试"""

    def test_migration_version_tracking(self, db):
        """测试版本号追踪"""
        mgr = MigrationManager(db.connection)
        version = mgr.get_current_version()
        assert version == 2  # v1 建表 + v2 索引

    def test_migration_history(self, db):
        """测试迁移历史"""
        mgr = MigrationManager(db.connection)
        history = mgr.get_migration_history()
        assert len(history) == 2
        assert history[0][0] == 1
        assert history[1][0] == 2

    def test_migration_idempotent(self, db):
        """测试迁移幂等性 (重复运行不报错)"""
        mgr = MigrationManager(db.connection)
        mgr.run_all()  # 不应报错
        version = mgr.get_current_version()
        assert version == 2

    def test_indexes_created(self, db):
        """测试索引已创建"""
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_%'"
        )
        index_names = {r[0] for r in rows}
        assert "idx_stock_daily_symbol" in index_names
        assert "idx_trade_record_date" in index_names
        assert "idx_backtest_result_strategy" in index_names


# ===========================================================================
# 9. AlertRepository 测试
# ===========================================================================


class TestAlertRepository:
    """告警仓库测试"""

    def test_add_and_get_active(self, alert_repo):
        """测试添加和查询活跃告警"""
        alert = Alert(
            symbol="000001",
            alert_type="price_above",
            threshold=15.0,
            message="价格突破15元",
        )
        aid = alert_repo.add(alert)
        assert aid > 0
        actives = alert_repo.get_active(symbol="000001")
        assert len(actives) == 1
        assert actives[0].threshold == 15.0

    def test_trigger_alert(self, alert_repo):
        """测试触发告警"""
        alert = Alert(
            symbol="000001",
            alert_type="price_above",
            threshold=15.0,
        )
        aid = alert_repo.add(alert)
        assert alert_repo.trigger(aid) is True

    def test_deactivate_alert(self, alert_repo):
        """测试停用告警"""
        alert = Alert(symbol="000001", alert_type="price_above", threshold=15.0)
        aid = alert_repo.add(alert)
        alert_repo.deactivate(aid)
        actives = alert_repo.get_active(symbol="000001")
        assert len(actives) == 0
