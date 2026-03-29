"""任务调度模块 - 定时任务管理

提供:
- TaskScheduler: 通用定时任务调度器
- QuantScheduler: 量化交易专用调度器（盘前/盘中/盘后任务）

用法::

    from quant_trading.scheduler import TaskScheduler, QuantScheduler

    # 通用调度器
    scheduler = TaskScheduler()
    scheduler.add_interval_task("fetch_data", my_func, interval=60)
    scheduler.start()

    # 量化专用调度器
    qs = QuantScheduler()
    qs.on_pre_market(my_pre_market_func)
    qs.on_market_open(my_market_func)
    qs.on_post_market(my_post_market_func)
    qs.start()
"""

from quant_trading.scheduler.task_scheduler import (
    TaskScheduler,
    QuantScheduler,
    ScheduledTask,
    TaskStatus,
)

__all__ = [
    "TaskScheduler",
    "QuantScheduler",
    "ScheduledTask",
    "TaskStatus",
]
