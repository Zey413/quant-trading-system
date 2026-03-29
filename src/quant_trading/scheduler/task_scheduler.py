"""任务调度器 - 定时行情、定时策略、盘前盘后

提供:
- ScheduledTask: 任务描述数据类
- TaskScheduler: 通用定时任务调度器（支持 interval / cron-like / 每日定时）
- QuantScheduler: 量化交易专用调度器（盘前/盘中/盘后 + A股交易时间感知）

用法::

    from quant_trading.scheduler import TaskScheduler

    scheduler = TaskScheduler()
    scheduler.add_interval_task("poll", fetch_data, interval=60)
    scheduler.add_daily_task("report", gen_report, hour=15, minute=30)
    scheduler.start()
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ======================================================================
# A股交易时间常量
# ======================================================================

# 盘前: 09:00 - 09:25 (集合竞价)
PRE_MARKET_START = (9, 0)
PRE_MARKET_END = (9, 25)

# 盘中: 09:30 - 11:30, 13:00 - 15:00
MORNING_OPEN = (9, 30)
MORNING_CLOSE = (11, 30)
AFTERNOON_OPEN = (13, 0)
AFTERNOON_CLOSE = (15, 0)

# 盘后: 15:00 - 15:30
POST_MARKET_START = (15, 0)
POST_MARKET_END = (15, 30)


def is_trading_day(dt: datetime | None = None) -> bool:
    """判断是否为交易日（简化版：周一到周五）

    注意：不包含节假日判断。生产环境应接入交易日历 API。

    Parameters
    ----------
    dt : datetime | None
        待判断的日期，默认当前日期

    Returns
    -------
    bool
        是否为交易日
    """
    dt = dt or datetime.now()
    return dt.weekday() < 5  # 0=Monday, 4=Friday


def is_trading_time(dt: datetime | None = None) -> bool:
    """判断是否在交易时间内

    Parameters
    ----------
    dt : datetime | None
        待判断的时间，默认当前时间

    Returns
    -------
    bool
        是否在交易时间
    """
    dt = dt or datetime.now()
    if not is_trading_day(dt):
        return False

    t = (dt.hour, dt.minute)
    morning = MORNING_OPEN <= t < MORNING_CLOSE
    afternoon = AFTERNOON_OPEN <= t < AFTERNOON_CLOSE
    return morning or afternoon


def is_pre_market_time(dt: datetime | None = None) -> bool:
    """判断是否在盘前时间"""
    dt = dt or datetime.now()
    if not is_trading_day(dt):
        return False
    t = (dt.hour, dt.minute)
    return PRE_MARKET_START <= t < PRE_MARKET_END


def is_post_market_time(dt: datetime | None = None) -> bool:
    """判断是否在盘后时间"""
    dt = dt or datetime.now()
    if not is_trading_day(dt):
        return False
    t = (dt.hour, dt.minute)
    return POST_MARKET_START <= t < POST_MARKET_END


# ======================================================================
# 任务模型
# ======================================================================


class TaskStatus(str, enum.Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, enum.Enum):
    """任务类型"""
    INTERVAL = "interval"    # 固定间隔
    DAILY = "daily"          # 每日定时
    ONCE = "once"            # 一次性
    TRADING_ONLY = "trading_only"  # 仅交易时间


@dataclass
class ScheduledTask:
    """计划任务描述

    Attributes
    ----------
    name : str
        任务名称（唯一标识）
    func : Callable
        执行函数
    task_type : TaskType
        任务类型
    interval : float
        间隔秒数（INTERVAL 类型）
    hour : int
        执行小时（DAILY 类型）
    minute : int
        执行分钟（DAILY 类型）
    args : tuple
        函数位置参数
    kwargs : dict
        函数关键字参数
    enabled : bool
        是否启用
    """
    name: str
    func: Callable
    task_type: TaskType = TaskType.INTERVAL
    interval: float = 60.0
    hour: int = 0
    minute: int = 0
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    enabled: bool = True

    # 运行时状态
    status: TaskStatus = TaskStatus.PENDING
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    last_error: str = ""

    def __post_init__(self) -> None:
        self._compute_next_run()

    def _compute_next_run(self) -> None:
        """计算下次执行时间"""
        now = datetime.now()
        if self.task_type == TaskType.INTERVAL:
            self.next_run = now + timedelta(seconds=self.interval)
        elif self.task_type == TaskType.DAILY:
            target = now.replace(
                hour=self.hour, minute=self.minute, second=0, microsecond=0
            )
            if target <= now:
                target += timedelta(days=1)
            self.next_run = target
        elif self.task_type == TaskType.ONCE:
            if self.next_run is None:
                self.next_run = now
        elif self.task_type == TaskType.TRADING_ONLY:
            self.next_run = now + timedelta(seconds=self.interval)

    def should_run(self, now: datetime | None = None) -> bool:
        """判断是否应该执行"""
        if not self.enabled:
            return False
        if self.status == TaskStatus.CANCELLED:
            return False

        now = now or datetime.now()

        if self.task_type == TaskType.TRADING_ONLY:
            if not is_trading_time(now):
                return False

        if self.next_run is not None and now >= self.next_run:
            return True

        return False

    def mark_run(self, success: bool, error: str = "") -> None:
        """标记执行结果"""
        self.last_run = datetime.now()
        self.run_count += 1
        if success:
            self.status = TaskStatus.SUCCESS
        else:
            self.status = TaskStatus.FAILED
            self.error_count += 1
            self.last_error = error

        # 计算下次执行时间
        if self.task_type == TaskType.ONCE:
            self.enabled = False
            self.next_run = None
        else:
            self._compute_next_run()


# ======================================================================
# 通用任务调度器
# ======================================================================


class TaskScheduler:
    """通用定时任务调度器

    支持:
    - 固定间隔任务 (interval)
    - 每日定时任务 (daily)
    - 一次性任务 (once)
    - 仅交易时间执行 (trading_only)
    - 任务启用/禁用
    - 运行状态监控

    Parameters
    ----------
    tick_interval : float
        调度循环的检查间隔（秒），默认 1.0

    Examples
    --------
    >>> scheduler = TaskScheduler()
    >>> scheduler.add_interval_task("poll", my_func, interval=60)
    >>> scheduler.start()
    >>> # ... later
    >>> scheduler.stop()
    """

    def __init__(self, tick_interval: float = 1.0) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running: bool = False
        self._tick_interval = tick_interval

        logger.debug("TaskScheduler 初始化: tick_interval=%.1f", tick_interval)

    # ------------------------------------------------------------------
    # 任务管理
    # ------------------------------------------------------------------

    def add_task(self, task: ScheduledTask) -> None:
        """添加任务

        Parameters
        ----------
        task : ScheduledTask
            任务实例

        Raises
        ------
        ValueError
            同名任务已存在
        """
        with self._lock:
            if task.name in self._tasks:
                raise ValueError(f"任务 {task.name!r} 已存在")
            self._tasks[task.name] = task
        logger.info("添加任务: %s (type=%s)", task.name, task.task_type.value)

    def add_interval_task(
        self,
        name: str,
        func: Callable,
        interval: float = 60.0,
        args: tuple = (),
        kwargs: dict | None = None,
        enabled: bool = True,
    ) -> ScheduledTask:
        """添加固定间隔任务

        Parameters
        ----------
        name : str
            任务名称
        func : Callable
            执行函数
        interval : float
            间隔秒数
        args : tuple
            函数位置参数
        kwargs : dict | None
            函数关键字参数
        enabled : bool
            是否启用

        Returns
        -------
        ScheduledTask
            创建的任务
        """
        task = ScheduledTask(
            name=name,
            func=func,
            task_type=TaskType.INTERVAL,
            interval=interval,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
        )
        self.add_task(task)
        return task

    def add_daily_task(
        self,
        name: str,
        func: Callable,
        hour: int = 0,
        minute: int = 0,
        args: tuple = (),
        kwargs: dict | None = None,
        enabled: bool = True,
    ) -> ScheduledTask:
        """添加每日定时任务

        Parameters
        ----------
        name : str
            任务名称
        func : Callable
            执行函数
        hour : int
            执行小时 (0-23)
        minute : int
            执行分钟 (0-59)

        Returns
        -------
        ScheduledTask
            创建的任务
        """
        task = ScheduledTask(
            name=name,
            func=func,
            task_type=TaskType.DAILY,
            hour=hour,
            minute=minute,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
        )
        self.add_task(task)
        return task

    def add_trading_task(
        self,
        name: str,
        func: Callable,
        interval: float = 60.0,
        args: tuple = (),
        kwargs: dict | None = None,
        enabled: bool = True,
    ) -> ScheduledTask:
        """添加仅交易时间执行的任务

        Parameters
        ----------
        name : str
            任务名称
        func : Callable
            执行函数
        interval : float
            间隔秒数（仅在交易时间内按此间隔执行）

        Returns
        -------
        ScheduledTask
            创建的任务
        """
        task = ScheduledTask(
            name=name,
            func=func,
            task_type=TaskType.TRADING_ONLY,
            interval=interval,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
        )
        self.add_task(task)
        return task

    def add_once_task(
        self,
        name: str,
        func: Callable,
        delay: float = 0.0,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> ScheduledTask:
        """添加一次性任务

        Parameters
        ----------
        name : str
            任务名称
        func : Callable
            执行函数
        delay : float
            延迟秒数

        Returns
        -------
        ScheduledTask
            创建的任务
        """
        task = ScheduledTask(
            name=name,
            func=func,
            task_type=TaskType.ONCE,
            args=args,
            kwargs=kwargs or {},
        )
        task.next_run = datetime.now() + timedelta(seconds=delay)
        self.add_task(task)
        return task

    def remove_task(self, name: str) -> bool:
        """移除任务

        Parameters
        ----------
        name : str
            任务名称

        Returns
        -------
        bool
            是否成功移除
        """
        with self._lock:
            if name in self._tasks:
                del self._tasks[name]
                logger.info("移除任务: %s", name)
                return True
        return False

    def enable_task(self, name: str) -> bool:
        """启用任务"""
        with self._lock:
            task = self._tasks.get(name)
            if task:
                task.enabled = True
                return True
        return False

    def disable_task(self, name: str) -> bool:
        """禁用任务"""
        with self._lock:
            task = self._tasks.get(name)
            if task:
                task.enabled = False
                return True
        return False

    def get_task(self, name: str) -> ScheduledTask | None:
        """获取任务"""
        with self._lock:
            return self._tasks.get(name)

    @property
    def tasks(self) -> list[str]:
        """所有任务名称"""
        with self._lock:
            return list(self._tasks.keys())

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def _execute_task(self, task: ScheduledTask) -> None:
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        logger.debug("执行任务: %s", task.name)

        try:
            task.func(*task.args, **task.kwargs)
            task.mark_run(success=True)
            logger.debug(
                "任务完成: %s (第 %d 次)", task.name, task.run_count
            )
        except Exception as exc:
            task.mark_run(success=False, error=str(exc))
            logger.error("任务失败: %s - %s", task.name, exc)

    def _scheduler_loop(self) -> None:
        """调度主循环"""
        logger.info("TaskScheduler 启动，共 %d 个任务", len(self._tasks))

        while not self._stop_event.is_set():
            now = datetime.now()

            with self._lock:
                tasks_to_run = [
                    t for t in self._tasks.values() if t.should_run(now)
                ]

            for task in tasks_to_run:
                self._execute_task(task)

            self._stop_event.wait(timeout=self._tick_interval)

        logger.info("TaskScheduler 已停止")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动调度器

        Raises
        ------
        RuntimeError
            已在运行中
        """
        if self._is_running:
            raise RuntimeError("TaskScheduler 已在运行中")

        self._stop_event.clear()
        self._is_running = True

        self._thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="task-scheduler",
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """停止调度器

        Parameters
        ----------
        timeout : float
            等待线程结束的超时秒数
        """
        if not self._is_running:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("调度线程未能在 %.1f 秒内停止", timeout)
            self._thread = None

        self._is_running = False
        logger.info("TaskScheduler 已停止")

    def run_task_now(self, name: str) -> bool:
        """立即执行指定任务（不影响调度计划）

        Parameters
        ----------
        name : str
            任务名称

        Returns
        -------
        bool
            任务是否存在并被执行
        """
        with self._lock:
            task = self._tasks.get(name)

        if task is None:
            logger.warning("任务不存在: %s", name)
            return False

        self._execute_task(task)
        return True

    @property
    def is_running(self) -> bool:
        """是否在运行中"""
        return self._is_running

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """获取调度器状态

        Returns
        -------
        dict[str, Any]
            调度器及所有任务的状态
        """
        with self._lock:
            task_statuses = {}
            for name, task in self._tasks.items():
                task_statuses[name] = {
                    "type": task.task_type.value,
                    "status": task.status.value,
                    "enabled": task.enabled,
                    "run_count": task.run_count,
                    "error_count": task.error_count,
                    "last_run": (
                        task.last_run.isoformat() if task.last_run else None
                    ),
                    "next_run": (
                        task.next_run.isoformat() if task.next_run else None
                    ),
                    "last_error": task.last_error,
                }

        return {
            "running": self._is_running,
            "task_count": len(task_statuses),
            "tasks": task_statuses,
        }

    def __repr__(self) -> str:
        return (
            f"TaskScheduler(tasks={len(self._tasks)}, "
            f"running={self._is_running})"
        )

    def __del__(self) -> None:
        if self._is_running:
            self.stop()


# ======================================================================
# 量化交易专用调度器
# ======================================================================


class QuantScheduler:
    """量化交易专用调度器

    封装 TaskScheduler，提供 A 股盘前/盘中/盘后任务的便捷注册接口。

    内置阶段:
    - pre_market: 盘前 (09:00)，执行数据准备、策略预计算
    - market_open: 开盘 (09:30)，执行开盘策略
    - market_poll: 盘中轮询，按 interval 定期执行（仅交易时间）
    - market_close: 收盘 (15:00)，执行收盘策略
    - post_market: 盘后 (15:05)，执行日报生成、数据归档

    Parameters
    ----------
    poll_interval : float
        盘中轮询间隔（秒），默认 60

    Examples
    --------
    >>> qs = QuantScheduler(poll_interval=30)
    >>> qs.on_pre_market(prepare_data)
    >>> qs.on_market_poll(check_signals)
    >>> qs.on_post_market(generate_report)
    >>> qs.start()
    """

    def __init__(self, poll_interval: float = 60.0) -> None:
        self._scheduler = TaskScheduler(tick_interval=1.0)
        self._poll_interval = poll_interval
        self._callbacks: dict[str, list[Callable]] = {
            "pre_market": [],
            "market_open": [],
            "market_poll": [],
            "market_close": [],
            "post_market": [],
        }
        logger.debug("QuantScheduler 初始化: poll_interval=%.1f", poll_interval)

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------

    def on_pre_market(self, func: Callable) -> Callable:
        """注册盘前回调（09:00 执行）

        Parameters
        ----------
        func : Callable
            回调函数

        Returns
        -------
        Callable
            返回回调本身，支持装饰器用法
        """
        self._callbacks["pre_market"].append(func)
        return func

    def on_market_open(self, func: Callable) -> Callable:
        """注册开盘回调（09:30 执行）"""
        self._callbacks["market_open"].append(func)
        return func

    def on_market_poll(self, func: Callable) -> Callable:
        """注册盘中轮询回调（仅交易时间内按 poll_interval 执行）"""
        self._callbacks["market_poll"].append(func)
        return func

    def on_market_close(self, func: Callable) -> Callable:
        """注册收盘回调（15:00 执行）"""
        self._callbacks["market_close"].append(func)
        return func

    def on_post_market(self, func: Callable) -> Callable:
        """注册盘后回调（15:05 执行）"""
        self._callbacks["post_market"].append(func)
        return func

    # ------------------------------------------------------------------
    # 内部：执行回调
    # ------------------------------------------------------------------

    def _run_callbacks(self, phase: str) -> None:
        """执行指定阶段的所有回调"""
        callbacks = self._callbacks.get(phase, [])
        logger.info("执行 %s 阶段回调，共 %d 个", phase, len(callbacks))
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                logger.error("%s 回调异常 [%s]: %s", phase, getattr(cb, "__name__", repr(cb)), exc)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动量化调度器

        自动注册所有阶段的任务到底层 TaskScheduler。
        """
        # 盘前: 09:00
        if self._callbacks["pre_market"]:
            self._scheduler.add_daily_task(
                name="quant_pre_market",
                func=self._run_callbacks,
                hour=PRE_MARKET_START[0],
                minute=PRE_MARKET_START[1],
                args=("pre_market",),
            )

        # 开盘: 09:30
        if self._callbacks["market_open"]:
            self._scheduler.add_daily_task(
                name="quant_market_open",
                func=self._run_callbacks,
                hour=MORNING_OPEN[0],
                minute=MORNING_OPEN[1],
                args=("market_open",),
            )

        # 盘中轮询
        if self._callbacks["market_poll"]:
            self._scheduler.add_trading_task(
                name="quant_market_poll",
                func=self._run_callbacks,
                interval=self._poll_interval,
                args=("market_poll",),
            )

        # 收盘: 15:00
        if self._callbacks["market_close"]:
            self._scheduler.add_daily_task(
                name="quant_market_close",
                func=self._run_callbacks,
                hour=AFTERNOON_CLOSE[0],
                minute=AFTERNOON_CLOSE[1],
                args=("market_close",),
            )

        # 盘后: 15:05
        if self._callbacks["post_market"]:
            self._scheduler.add_daily_task(
                name="quant_post_market",
                func=self._run_callbacks,
                hour=15,
                minute=5,
                args=("post_market",),
            )

        self._scheduler.start()
        logger.info("QuantScheduler 已启动")

    def stop(self) -> None:
        """停止量化调度器"""
        self._scheduler.stop()
        logger.info("QuantScheduler 已停止")

    @property
    def is_running(self) -> bool:
        """是否在运行中"""
        return self._scheduler.is_running

    def get_status(self) -> dict[str, Any]:
        """获取调度器状态"""
        status = self._scheduler.get_status()
        status["callbacks"] = {
            phase: len(cbs) for phase, cbs in self._callbacks.items()
        }
        status["is_trading_day"] = is_trading_day()
        status["is_trading_time"] = is_trading_time()
        return status

    def run_phase(self, phase: str) -> None:
        """手动触发指定阶段的回调（用于测试/调试）

        Parameters
        ----------
        phase : str
            阶段名称: pre_market / market_open / market_poll /
            market_close / post_market
        """
        if phase not in self._callbacks:
            raise ValueError(
                f"未知阶段: {phase!r}，可选: {list(self._callbacks.keys())}"
            )
        self._run_callbacks(phase)

    def __repr__(self) -> str:
        cb_counts = {k: len(v) for k, v in self._callbacks.items() if v}
        return (
            f"QuantScheduler(poll_interval={self._poll_interval}, "
            f"callbacks={cb_counts}, running={self.is_running})"
        )
