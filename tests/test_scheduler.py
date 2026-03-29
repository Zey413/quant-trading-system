"""任务调度模块单元测试

测试覆盖:
- 交易时间判断工具函数
- ScheduledTask 数据类行为
- TaskScheduler 任务增删、启停、执行
- QuantScheduler 阶段回调注册与触发
- 不依赖真实时间等待（通过 mock 控制时间）
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from quant_trading.scheduler.task_scheduler import (
    TaskScheduler,
    QuantScheduler,
    ScheduledTask,
    TaskStatus,
    TaskType,
    is_trading_day,
    is_trading_time,
    is_pre_market_time,
    is_post_market_time,
)


# ======================================================================
# Tests: 交易时间工具函数
# ======================================================================


class TestTradingTimeUtils:
    """交易时间工具函数测试"""

    def test_weekday_is_trading_day(self):
        """工作日是交易日"""
        # 2026-03-30 是周一
        monday = datetime(2026, 3, 30, 10, 0)
        assert is_trading_day(monday) is True

    def test_weekend_is_not_trading_day(self):
        """周末不是交易日"""
        # 2026-03-28 是周六
        saturday = datetime(2026, 3, 28, 10, 0)
        assert is_trading_day(saturday) is False

        # 2026-03-29 是周日
        sunday = datetime(2026, 3, 29, 10, 0)
        assert is_trading_day(sunday) is False

    def test_morning_trading_time(self):
        """上午交易时间"""
        dt = datetime(2026, 3, 30, 10, 0)  # 周一 10:00
        assert is_trading_time(dt) is True

    def test_afternoon_trading_time(self):
        """下午交易时间"""
        dt = datetime(2026, 3, 30, 14, 0)  # 周一 14:00
        assert is_trading_time(dt) is True

    def test_lunch_break_not_trading(self):
        """午休时间不是交易时间"""
        dt = datetime(2026, 3, 30, 12, 0)  # 周一 12:00
        assert is_trading_time(dt) is False

    def test_before_market_not_trading(self):
        """开盘前不是交易时间"""
        dt = datetime(2026, 3, 30, 9, 0)  # 周一 09:00
        assert is_trading_time(dt) is False

    def test_after_market_not_trading(self):
        """收盘后不是交易时间"""
        dt = datetime(2026, 3, 30, 15, 30)  # 周一 15:30
        assert is_trading_time(dt) is False

    def test_weekend_not_trading_time(self):
        """周末即使在正常交易时间段也不算交易时间"""
        dt = datetime(2026, 3, 28, 10, 0)  # 周六 10:00
        assert is_trading_time(dt) is False

    def test_pre_market_time(self):
        """盘前时间"""
        dt = datetime(2026, 3, 30, 9, 10)  # 周一 09:10
        assert is_pre_market_time(dt) is True

    def test_not_pre_market_time(self):
        """非盘前时间"""
        dt = datetime(2026, 3, 30, 10, 0)
        assert is_pre_market_time(dt) is False

    def test_post_market_time(self):
        """盘后时间"""
        dt = datetime(2026, 3, 30, 15, 10)  # 周一 15:10
        assert is_post_market_time(dt) is True

    def test_not_post_market_time(self):
        """非盘后时间"""
        dt = datetime(2026, 3, 30, 14, 0)
        assert is_post_market_time(dt) is False


# ======================================================================
# Tests: ScheduledTask
# ======================================================================


class TestScheduledTask:
    """ScheduledTask 数据类测试"""

    def test_create_interval_task(self):
        """创建间隔任务"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.INTERVAL,
            interval=30.0,
        )
        assert task.name == "test"
        assert task.task_type == TaskType.INTERVAL
        assert task.interval == 30.0
        assert task.enabled is True
        assert task.status == TaskStatus.PENDING
        assert task.next_run is not None

    def test_create_daily_task(self):
        """创建每日任务"""
        task = ScheduledTask(
            name="daily_report",
            func=lambda: None,
            task_type=TaskType.DAILY,
            hour=15,
            minute=30,
        )
        assert task.hour == 15
        assert task.minute == 30
        assert task.next_run is not None

    def test_should_run_when_due(self):
        """到期时应该执行"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.INTERVAL,
            interval=1.0,
        )
        # 模拟已到期
        task.next_run = datetime.now() - timedelta(seconds=1)
        assert task.should_run() is True

    def test_should_not_run_before_due(self):
        """未到期时不应执行"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.INTERVAL,
            interval=3600.0,
        )
        # next_run 是未来
        assert task.should_run() is False

    def test_should_not_run_when_disabled(self):
        """禁用时不应执行"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.INTERVAL,
            interval=1.0,
            enabled=False,
        )
        task.next_run = datetime.now() - timedelta(seconds=1)
        assert task.should_run() is False

    def test_should_not_run_when_cancelled(self):
        """取消时不应执行"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.INTERVAL,
            interval=1.0,
        )
        task.status = TaskStatus.CANCELLED
        task.next_run = datetime.now() - timedelta(seconds=1)
        assert task.should_run() is False

    def test_trading_only_requires_trading_time(self):
        """TRADING_ONLY 类型需要在交易时间内"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.TRADING_ONLY,
            interval=1.0,
        )
        task.next_run = datetime.now() - timedelta(seconds=1)

        # 周末不执行
        weekend = datetime(2026, 3, 28, 10, 0)
        assert task.should_run(weekend) is False

        # 交易时间内执行
        trading = datetime(2026, 3, 30, 10, 0)
        task.next_run = trading - timedelta(seconds=1)
        assert task.should_run(trading) is True

    def test_mark_run_success(self):
        """标记成功执行"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.INTERVAL,
            interval=60.0,
        )
        task.mark_run(success=True)
        assert task.status == TaskStatus.SUCCESS
        assert task.run_count == 1
        assert task.error_count == 0
        assert task.last_run is not None

    def test_mark_run_failure(self):
        """标记失败执行"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.INTERVAL,
            interval=60.0,
        )
        task.mark_run(success=False, error="network error")
        assert task.status == TaskStatus.FAILED
        assert task.run_count == 1
        assert task.error_count == 1
        assert task.last_error == "network error"

    def test_once_task_disables_after_run(self):
        """一次性任务执行后自动禁用"""
        task = ScheduledTask(
            name="test",
            func=lambda: None,
            task_type=TaskType.ONCE,
        )
        task.mark_run(success=True)
        assert task.enabled is False
        assert task.next_run is None


# ======================================================================
# Tests: TaskScheduler
# ======================================================================


class TestTaskScheduler:
    """TaskScheduler 测试"""

    def test_add_interval_task(self):
        """添加间隔任务"""
        scheduler = TaskScheduler()
        mock_func = MagicMock()
        task = scheduler.add_interval_task("poll", mock_func, interval=30)
        assert "poll" in scheduler.tasks
        assert task.task_type == TaskType.INTERVAL

    def test_add_daily_task(self):
        """添加每日任务"""
        scheduler = TaskScheduler()
        mock_func = MagicMock()
        task = scheduler.add_daily_task("report", mock_func, hour=15, minute=30)
        assert "report" in scheduler.tasks
        assert task.task_type == TaskType.DAILY

    def test_add_trading_task(self):
        """添加交易时间任务"""
        scheduler = TaskScheduler()
        mock_func = MagicMock()
        task = scheduler.add_trading_task("check", mock_func, interval=60)
        assert "check" in scheduler.tasks
        assert task.task_type == TaskType.TRADING_ONLY

    def test_add_once_task(self):
        """添加一次性任务"""
        scheduler = TaskScheduler()
        mock_func = MagicMock()
        task = scheduler.add_once_task("init", mock_func, delay=0)
        assert "init" in scheduler.tasks
        assert task.task_type == TaskType.ONCE

    def test_add_duplicate_task(self):
        """重复添加应抛出 ValueError"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None)
        with pytest.raises(ValueError, match="已存在"):
            scheduler.add_interval_task("poll", lambda: None)

    def test_remove_task(self):
        """移除任务"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None)
        assert scheduler.remove_task("poll") is True
        assert "poll" not in scheduler.tasks

    def test_remove_nonexistent_task(self):
        """移除不存在的任务"""
        scheduler = TaskScheduler()
        assert scheduler.remove_task("xxx") is False

    def test_enable_disable_task(self):
        """启用/禁用任务"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None)
        assert scheduler.disable_task("poll") is True
        task = scheduler.get_task("poll")
        assert task is not None
        assert task.enabled is False

        assert scheduler.enable_task("poll") is True
        assert task.enabled is True

    def test_get_task(self):
        """获取任务"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None)
        task = scheduler.get_task("poll")
        assert task is not None
        assert task.name == "poll"

        assert scheduler.get_task("xxx") is None

    def test_start_stop(self):
        """启动和停止"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None, interval=100)
        scheduler.start()
        assert scheduler.is_running is True

        scheduler.stop()
        assert scheduler.is_running is False

    def test_double_start(self):
        """重复启动应抛出 RuntimeError"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None, interval=100)
        scheduler.start()
        try:
            with pytest.raises(RuntimeError, match="已在运行中"):
                scheduler.start()
        finally:
            scheduler.stop()

    def test_stop_when_not_running(self):
        """未运行时停止不应报错"""
        scheduler = TaskScheduler()
        scheduler.stop()  # 不抛异常

    def test_task_execution(self):
        """任务实际执行"""
        scheduler = TaskScheduler(tick_interval=0.05)
        mock_func = MagicMock()

        task = scheduler.add_interval_task("poll", mock_func, interval=0.01)
        # 手动设置 next_run 为过去
        task.next_run = datetime.now() - timedelta(seconds=1)

        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()

        assert mock_func.call_count >= 1

    def test_task_with_args(self):
        """带参数的任务执行"""
        scheduler = TaskScheduler(tick_interval=0.05)
        mock_func = MagicMock()

        task = scheduler.add_interval_task(
            "poll", mock_func,
            interval=0.01,
            args=("arg1",),
            kwargs={"key1": "val1"},
        )
        task.next_run = datetime.now() - timedelta(seconds=1)

        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()

        mock_func.assert_called_with("arg1", key1="val1")

    def test_task_failure_handled(self):
        """任务失败不影响调度器"""
        scheduler = TaskScheduler(tick_interval=0.05)
        mock_func = MagicMock(side_effect=RuntimeError("boom"))

        task = scheduler.add_interval_task("fail", mock_func, interval=0.01)
        task.next_run = datetime.now() - timedelta(seconds=1)

        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()

        # 任务失败但调度器继续运行
        assert task.error_count >= 1
        assert task.status == TaskStatus.FAILED

    def test_run_task_now(self):
        """立即执行任务"""
        scheduler = TaskScheduler()
        mock_func = MagicMock()
        scheduler.add_interval_task("poll", mock_func, interval=3600)

        result = scheduler.run_task_now("poll")
        assert result is True
        mock_func.assert_called_once()

    def test_run_task_now_nonexistent(self):
        """立即执行不存在的任务"""
        scheduler = TaskScheduler()
        result = scheduler.run_task_now("xxx")
        assert result is False

    def test_get_status(self):
        """获取调度器状态"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None, interval=60)

        status = scheduler.get_status()
        assert status["running"] is False
        assert status["task_count"] == 1
        assert "poll" in status["tasks"]
        assert status["tasks"]["poll"]["type"] == "interval"

    def test_repr(self):
        """repr 格式"""
        scheduler = TaskScheduler()
        scheduler.add_interval_task("poll", lambda: None)
        r = repr(scheduler)
        assert "TaskScheduler" in r
        assert "tasks=1" in r


# ======================================================================
# Tests: QuantScheduler
# ======================================================================


class TestQuantScheduler:
    """QuantScheduler 测试"""

    def test_register_pre_market(self):
        """注册盘前回调"""
        qs = QuantScheduler()
        mock_func = MagicMock()
        result = qs.on_pre_market(mock_func)
        assert result is mock_func

    def test_register_market_open(self):
        """注册开盘回调"""
        qs = QuantScheduler()
        mock_func = MagicMock()
        qs.on_market_open(mock_func)

    def test_register_market_poll(self):
        """注册盘中轮询回调"""
        qs = QuantScheduler()
        mock_func = MagicMock()
        qs.on_market_poll(mock_func)

    def test_register_market_close(self):
        """注册收盘回调"""
        qs = QuantScheduler()
        mock_func = MagicMock()
        qs.on_market_close(mock_func)

    def test_register_post_market(self):
        """注册盘后回调"""
        qs = QuantScheduler()
        mock_func = MagicMock()
        qs.on_post_market(mock_func)

    def test_run_phase_pre_market(self):
        """手动触发盘前阶段"""
        qs = QuantScheduler()
        mock1 = MagicMock()
        mock2 = MagicMock()
        qs.on_pre_market(mock1)
        qs.on_pre_market(mock2)

        qs.run_phase("pre_market")
        mock1.assert_called_once()
        mock2.assert_called_once()

    def test_run_phase_post_market(self):
        """手动触发盘后阶段"""
        qs = QuantScheduler()
        mock_func = MagicMock()
        qs.on_post_market(mock_func)

        qs.run_phase("post_market")
        mock_func.assert_called_once()

    def test_run_phase_unknown(self):
        """触发未知阶段应抛出 ValueError"""
        qs = QuantScheduler()
        with pytest.raises(ValueError, match="未知阶段"):
            qs.run_phase("unknown_phase")

    def test_callback_error_handled(self):
        """回调异常不影响其他回调"""
        qs = QuantScheduler()
        mock_fail = MagicMock(side_effect=RuntimeError("fail"))
        mock_success = MagicMock()
        qs.on_pre_market(mock_fail)
        qs.on_pre_market(mock_success)

        qs.run_phase("pre_market")
        mock_fail.assert_called_once()
        mock_success.assert_called_once()

    def test_start_stop(self):
        """启动和停止"""
        qs = QuantScheduler()
        qs.on_pre_market(lambda: None)
        qs.start()
        assert qs.is_running is True

        qs.stop()
        assert qs.is_running is False

    def test_get_status(self):
        """获取状态"""
        qs = QuantScheduler()
        qs.on_pre_market(lambda: None)
        qs.on_post_market(lambda: None)

        status = qs.get_status()
        assert "callbacks" in status
        assert status["callbacks"]["pre_market"] == 1
        assert status["callbacks"]["post_market"] == 1
        assert "is_trading_day" in status
        assert "is_trading_time" in status

    def test_decorator_usage(self):
        """装饰器风格注册"""
        qs = QuantScheduler()

        @qs.on_pre_market
        def my_pre_market():
            pass

        @qs.on_post_market
        def my_post_market():
            pass

        assert len(qs._callbacks["pre_market"]) == 1
        assert len(qs._callbacks["post_market"]) == 1

    def test_repr(self):
        """repr 格式"""
        qs = QuantScheduler(poll_interval=30)
        qs.on_pre_market(lambda: None)
        r = repr(qs)
        assert "QuantScheduler" in r
        assert "poll_interval=30" in r

    def test_multiple_callbacks_per_phase(self):
        """每个阶段可注册多个回调"""
        qs = QuantScheduler()
        mocks = [MagicMock() for _ in range(5)]
        for m in mocks:
            qs.on_market_poll(m)

        qs.run_phase("market_poll")
        for m in mocks:
            m.assert_called_once()
