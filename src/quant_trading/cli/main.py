"""命令行工具 - 基于Click的量化交易系统CLI

提供以下子命令:
- fetch:      获取股票行情数据
- backtest:   运行策略回测
- strategies: 列出所有可用策略
- plot:       生成可视化图表

使用Rich美化终端输出。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# 日志配置
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  辅助函数
# ------------------------------------------------------------------ #

def _load_config(config_path: str):
    """加载应用配置，失败时提供友好提示。"""
    try:
        from quant_trading.core.config import AppConfig
        return AppConfig.from_yaml(config_path)
    except Exception as e:
        console.print(f"[bold red]错误:[/] 无法加载配置文件 '{config_path}': {e}")
        sys.exit(1)


def _format_number(value: float, precision: int = 2) -> str:
    """格式化数字，千分位分隔。"""
    if abs(value) >= 1_000_000:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"


def _colored_pct(value: float) -> str:
    """根据正负值返回带颜色的百分比字符串 (中国标准: 红涨绿跌)。"""
    pct_str = f"{value:.2%}"
    if value > 0:
        return f"[bold red]+{pct_str}[/]"
    elif value < 0:
        return f"[bold green]{pct_str}[/]"
    return pct_str


# ------------------------------------------------------------------ #
#  主命令组
# ------------------------------------------------------------------ #

@click.group()
@click.option(
    '--config', '-c',
    default='config/default.yaml',
    help='配置文件路径',
    type=click.Path(),
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    default=False,
    help='详细日志输出',
)
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """量化交易系统 - A股量化分析工具

    \b
    支持的功能:
      • 获取A股行情数据 (akshare/tushare)
      • 运行策略回测 (MA交叉/RSI/MACD/布林带/复合)
      • 生成可视化图表 (K线/信号/净值/回撤)

    \b
    示例:
      quant fetch -s 000001 --start 2024-01-01
      quant backtest -st ma_crossover -s 600519
      quant strategies
      quant plot -s 000001 -st ma_crossover
    """
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['config'] = _load_config(config)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# ------------------------------------------------------------------ #
#  fetch - 获取股票数据
# ------------------------------------------------------------------ #

@cli.command()
@click.option('--symbol', '-s', required=True, help='股票代码 (如: 000001)')
@click.option('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
@click.option('--end', default='2025-12-31', help='结束日期 (YYYY-MM-DD)')
@click.option('--source', default=None, help='数据源 (akshare/tushare)')
@click.option('--rows', '-n', default=20, help='显示行数 (默认20, 0=全部)')
@click.pass_context
def fetch(ctx: click.Context, symbol: str, start: str, end: str,
          source: str | None, rows: int) -> None:
    """获取股票数据

    \b
    示例:
      quant fetch -s 000001 --start 2024-01-01 --end 2024-12-31
      quant fetch -s 600519 --source akshare -n 50
    """
    config = ctx.obj['config']

    try:
        from quant_trading.data.manager import DataManager
    except ImportError as e:
        console.print(f"[bold red]错误:[/] 无法导入数据模块: {e}")
        sys.exit(1)

    dm = DataManager(config)

    # 切换数据源
    if source:
        try:
            dm.set_source(source)
        except ValueError as e:
            console.print(f"[bold red]错误:[/] {e}")
            sys.exit(1)

    console.print(
        f"\n[bold cyan]正在获取数据...[/]  "
        f"股票: [bold]{symbol}[/]  "
        f"日期: {start} ~ {end}  "
        f"数据源: {dm.source_name}"
    )

    try:
        df = dm.fetch_daily(symbol, start, end)
    except Exception as e:
        console.print(f"[bold red]获取数据失败:[/] {e}")
        sys.exit(1)

    if df.empty:
        console.print("[yellow]未获取到数据，请检查股票代码和日期范围。[/]")
        return

    # 构建Rich表格
    table = Table(
        title=f"📊 {symbol} 日线数据 ({start} ~ {end})",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("日期", style="white", justify="center")
    table.add_column("开盘", style="white", justify="right")
    table.add_column("最高", style="white", justify="right")
    table.add_column("最低", style="white", justify="right")
    table.add_column("收盘", style="white", justify="right")
    table.add_column("涨跌", justify="right")
    table.add_column("成交量", style="dim", justify="right")

    # 确定显示范围
    display_df = df if (rows == 0 or len(df) <= rows) else df.tail(rows)

    for _, row in display_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        change = row['close'] - row['open']
        change_pct = change / row['open'] if row['open'] != 0 else 0

        # 涨跌颜色 (中国标准)
        if change > 0:
            change_str = f"[bold red]+{change:.2f} ({change_pct:+.2%})[/]"
        elif change < 0:
            change_str = f"[bold green]{change:.2f} ({change_pct:+.2%})[/]"
        else:
            change_str = f"{change:.2f} ({change_pct:+.2%})"

        vol_str = f"{int(row['volume']):,}"

        table.add_row(
            date_str,
            f"{row['open']:.2f}",
            f"{row['high']:.2f}",
            f"{row['low']:.2f}",
            f"{row['close']:.2f}",
            change_str,
            vol_str,
        )

    console.print()
    console.print(table)
    console.print(
        f"\n  共 [bold]{len(df)}[/] 条数据"
        + (f"  (显示最近 {rows} 条)" if rows > 0 and len(df) > rows else "")
    )


# ------------------------------------------------------------------ #
#  backtest - 运行回测
# ------------------------------------------------------------------ #

@cli.command()
@click.option('--strategy', '-st', required=True, help='策略名称 (如: ma_crossover)')
@click.option('--symbol', '-s', required=True, help='股票代码 (如: 000001)')
@click.option('--start', default=None, help='开始日期 (默认使用配置)')
@click.option('--end', default=None, help='结束日期 (默认使用配置)')
@click.option('--capital', default=None, type=float, help='初始资金 (默认使用配置)')
@click.option('--save', default=None, help='保存回测图表路径')
@click.pass_context
def backtest(ctx: click.Context, strategy: str, symbol: str,
             start: str | None, end: str | None, capital: float | None,
             save: str | None) -> None:
    """运行策略回测

    \b
    示例:
      quant backtest -st ma_crossover -s 000001
      quant backtest -st rsi -s 600519 --capital 500000
      quant backtest -st macd -s 000001 --start 2023-01-01 --end 2024-12-31 --save report.png
    """
    config = ctx.obj['config']

    # 覆盖配置中的日期和资金
    start_date = start or config.backtest.start_date
    end_date = end or config.backtest.end_date
    if capital is not None:
        config.backtest.initial_capital = capital

    # 导入依赖模块
    try:
        from quant_trading.data.manager import DataManager
        from quant_trading.strategy import StrategyRegistry
        from quant_trading.backtest.engine import BacktestEngine
    except ImportError as e:
        console.print(f"[bold red]错误:[/] 无法导入模块: {e}")
        sys.exit(1)

    # 获取策略
    try:
        available = StrategyRegistry.list_strategies()
        strategy_instance = StrategyRegistry.get(strategy)
    except (KeyError, ValueError) as e:
        console.print(f"[bold red]错误:[/] {e}")
        console.print(f"[dim]可用策略: {', '.join(StrategyRegistry.list_strategies())}[/]")
        sys.exit(1)

    console.print(Panel(
        f"[bold]策略:[/] {strategy}  |  "
        f"[bold]股票:[/] {symbol}  |  "
        f"[bold]日期:[/] {start_date} ~ {end_date}  |  "
        f"[bold]初始资金:[/] {_format_number(config.backtest.initial_capital)}元",
        title="[bold cyan]回测参数[/]",
        border_style="cyan",
    ))

    # 获取数据
    console.print("\n[dim]正在获取行情数据...[/]")
    dm = DataManager(config)
    try:
        data = dm.fetch_daily(symbol, start_date, end_date)
    except Exception as e:
        console.print(f"[bold red]获取数据失败:[/] {e}")
        sys.exit(1)

    if data.empty:
        console.print("[yellow]未获取到数据，请检查股票代码和日期范围。[/]")
        return

    console.print(f"[dim]获取到 {len(data)} 条行情数据[/]")

    # 运行回测
    console.print("[dim]正在运行回测...[/]")
    engine = BacktestEngine(config)

    try:
        result = engine.run(strategy_instance, data, symbol=symbol)
    except Exception as e:
        console.print(f"[bold red]回测执行失败:[/] {e}")
        logger.exception("回测执行失败")
        sys.exit(1)

    # 展示结果
    _display_backtest_result(result, strategy, symbol)

    # 生成图表
    if save:
        try:
            from quant_trading.visualization.charts import ChartGenerator
            # 生成带信号的数据
            signal_data = strategy_instance.generate_signals(data.copy())
            ChartGenerator.plot_backtest_report(
                data=signal_data,
                equity_curve=result.equity_curve,
                title=f"{symbol} - {strategy} 回测报告",
                save_path=save,
            )
            console.print(f"\n[bold green]图表已保存至:[/] {save}")
        except Exception as e:
            console.print(f"[yellow]图表生成失败:[/] {e}")


def _display_backtest_result(result, strategy_name: str, symbol: str) -> None:
    """使用Rich表格展示回测结果。"""
    console.print()

    # 收益指标表
    returns_table = Table(
        title="📈 收益指标",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold yellow",
    )
    returns_table.add_column("指标", style="white", justify="left")
    returns_table.add_column("数值", justify="right")
    returns_table.add_row("总收益率", _colored_pct(result.total_return))
    returns_table.add_row("年化收益率", _colored_pct(result.annualized_return))
    console.print(returns_table)

    # 风险指标表
    risk_table = Table(
        title="⚠️  风险指标",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold yellow",
    )
    risk_table.add_column("指标", style="white", justify="left")
    risk_table.add_column("数值", justify="right")
    risk_table.add_row("最大回撤", f"[bold red]{result.max_drawdown:.2%}[/]")
    risk_table.add_row("最大回撤天数", f"{result.max_drawdown_duration} 天")
    risk_table.add_row("年化波动率", f"{result.volatility:.2%}")
    console.print(risk_table)

    # 风险调整收益表
    ra_table = Table(
        title="📊 风险调整收益",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold yellow",
    )
    ra_table.add_column("指标", style="white", justify="left")
    ra_table.add_column("数值", justify="right")

    # 夏普比率颜色
    sharpe_color = "green" if result.sharpe_ratio > 1 else ("yellow" if result.sharpe_ratio > 0 else "red")
    ra_table.add_row("夏普比率", f"[bold {sharpe_color}]{result.sharpe_ratio:.4f}[/]")
    ra_table.add_row("Sortino比率", f"{result.sortino_ratio:.4f}")
    ra_table.add_row("Calmar比率", f"{result.calmar_ratio:.4f}")
    console.print(ra_table)

    # 交易统计表
    trade_table = Table(
        title="🔄 交易统计",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold yellow",
    )
    trade_table.add_column("指标", style="white", justify="left")
    trade_table.add_column("数值", justify="right")
    trade_table.add_row("总交易次数", f"{result.total_trades}")
    trade_table.add_row("胜率", f"{result.win_rate:.2%}")
    trade_table.add_row("盈亏比", f"{result.profit_loss_ratio:.4f}")
    trade_table.add_row("平均盈利", f"[red]{_format_number(result.avg_win)}[/] 元")
    trade_table.add_row("平均亏损", f"[green]{_format_number(result.avg_loss)}[/] 元")
    console.print(trade_table)


# ------------------------------------------------------------------ #
#  strategies - 列出策略
# ------------------------------------------------------------------ #

@cli.command()
def strategies() -> None:
    """列出所有可用策略

    \b
    显示所有已注册的交易策略及其参数信息。
    """
    try:
        from quant_trading.strategy import StrategyRegistry
    except ImportError as e:
        console.print(f"[bold red]错误:[/] 无法导入策略模块: {e}")
        sys.exit(1)

    strategy_names = StrategyRegistry.list_strategies()

    if not strategy_names:
        console.print("[yellow]没有已注册的策略。[/]")
        return

    console.print(Panel(
        f"共 [bold]{len(strategy_names)}[/] 个可用策略",
        title="[bold cyan]策略列表[/]",
        border_style="cyan",
    ))

    for name in strategy_names:
        try:
            # 创建实例获取参数信息
            # composite策略需要子策略参数，单独处理
            if name == "composite":
                console.print(f"  [bold]📋 {name}[/]  [dim](复合策略 — 需传入子策略列表)[/]")
                console.print(f"     参数: strategies (子策略列表), voting (majority/unanimous/any)")
                console.print()
                continue

            instance = StrategyRegistry.get(name)
            params = instance.get_params()

            table = Table(
                title=f"📋 {name}",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                title_style="bold white",
            )
            table.add_column("参数", style="white")
            table.add_column("默认值", style="yellow", justify="right")

            if params:
                for k, v in params.items():
                    table.add_row(k, str(v))
            else:
                table.add_row("[dim]无参数[/]", "-")

            console.print(table)
            console.print()

        except Exception as e:
            console.print(f"  [bold]{name}[/]  [dim red](加载失败: {e})[/]")
            console.print()


# ------------------------------------------------------------------ #
#  plot - 生成图表
# ------------------------------------------------------------------ #

@cli.command()
@click.option('--symbol', '-s', required=True, help='股票代码 (如: 000001)')
@click.option('--strategy', '-st', default=None, help='策略名称 (可选, 显示信号)')
@click.option('--start', default='2024-01-01', help='开始日期')
@click.option('--end', default='2025-12-31', help='结束日期')
@click.option(
    '--chart-type', '-t',
    type=click.Choice(['candlestick', 'signals', 'all'], case_sensitive=False),
    default='all',
    help='图表类型 (默认: all)',
)
@click.option('--save', default=None, help='保存图片路径')
@click.pass_context
def plot(ctx: click.Context, symbol: str, strategy: str | None,
         start: str, end: str, chart_type: str, save: str | None) -> None:
    """生成可视化图表

    \b
    图表类型:
      candlestick  - K线图
      signals      - 交易信号图 (需指定策略)
      all          - 全部图表

    \b
    示例:
      quant plot -s 000001 -t candlestick
      quant plot -s 600519 -st ma_crossover --save signals.png
      quant plot -s 000001 -st rsi -t all
    """
    config = ctx.obj['config']

    try:
        from quant_trading.data.manager import DataManager
        from quant_trading.visualization.charts import ChartGenerator
    except ImportError as e:
        console.print(f"[bold red]错误:[/] 无法导入模块: {e}")
        sys.exit(1)

    # 获取数据
    console.print(f"\n[dim]正在获取 {symbol} 行情数据 ({start} ~ {end})...[/]")
    dm = DataManager(config)
    try:
        data = dm.fetch_daily(symbol, start, end)
    except Exception as e:
        console.print(f"[bold red]获取数据失败:[/] {e}")
        sys.exit(1)

    if data.empty:
        console.print("[yellow]未获取到数据，请检查股票代码和日期范围。[/]")
        return

    console.print(f"[dim]获取到 {len(data)} 条行情数据[/]")

    # 如果指定了策略，生成信号
    signal_data = None
    if strategy:
        try:
            from quant_trading.strategy import StrategyRegistry
            strategy_instance = StrategyRegistry.get(strategy)
            signal_data = strategy_instance.generate_signals(data.copy())
            console.print(f"[dim]已使用 {strategy} 策略生成交易信号[/]")
        except (KeyError, ValueError) as e:
            console.print(f"[yellow]策略加载失败: {e}[/]")
        except Exception as e:
            console.print(f"[yellow]信号生成失败: {e}[/]")

    # 生成图表
    plot_data = signal_data if signal_data is not None else data
    title_prefix = f"{symbol}"
    if strategy:
        title_prefix += f" ({strategy})"

    if chart_type in ('candlestick', 'all'):
        console.print("[dim]正在生成K线图...[/]")
        kline_save = None
        if save and chart_type == 'candlestick':
            kline_save = save
        elif save and chart_type == 'all':
            # 为all模式下的K线图生成带后缀的文件名
            base, ext = _split_save_path(save)
            kline_save = f"{base}_candlestick{ext}"
        ChartGenerator.plot_candlestick(
            plot_data, title=f"{title_prefix} K线图", save_path=kline_save,
        )
        if kline_save:
            console.print(f"[green]  已保存: {kline_save}[/]")

    if chart_type in ('signals', 'all'):
        if signal_data is not None:
            console.print("[dim]正在生成交易信号图...[/]")
            sig_save = None
            if save and chart_type == 'signals':
                sig_save = save
            elif save and chart_type == 'all':
                base, ext = _split_save_path(save)
                sig_save = f"{base}_signals{ext}"
            ChartGenerator.plot_signals(
                signal_data, title=f"{title_prefix} 交易信号图", save_path=sig_save,
            )
            if sig_save:
                console.print(f"[green]  已保存: {sig_save}[/]")
        elif chart_type == 'signals':
            console.print("[yellow]未指定策略 (-st)，无法生成交易信号图。[/]")

    console.print("\n[bold green]图表生成完成![/]")


def _split_save_path(save_path: str) -> tuple[str, str]:
    """拆分保存路径为 (basename, extension)。"""
    import os
    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = '.png'
    return base, ext


# ------------------------------------------------------------------ #
#  入口
# ------------------------------------------------------------------ #

def main() -> None:
    """CLI入口函数"""
    cli(obj={})


if __name__ == '__main__':
    main()
