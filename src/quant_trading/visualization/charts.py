"""图表生成器 - 基于matplotlib的量化交易可视化工具

提供:
- K线图（简化版，红涨绿跌 - 中国标准）
- 交易信号图（买卖点标注）
- 净值曲线图（含回撤区域填充）
- 回撤图
- 月度收益热力图

所有图表均支持中文显示和保存至文件。
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# 中文字体配置 (兼容 macOS / Windows / Linux)
# ------------------------------------------------------------------ #
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei',            # Windows
    'Arial Unicode MS',  # macOS
    'WenQuanYi Micro Hei',  # Linux
    'DejaVu Sans',       # 兜底
]
matplotlib.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# 配色方案 — 中国股市标准: 红涨绿跌
# ------------------------------------------------------------------ #
COLOR_UP = '#EF5350'       # 阳线 — 红色
COLOR_DOWN = '#26A69A'     # 阴线 — 绿色
COLOR_BUY = '#EF5350'      # 买入信号 — 红色
COLOR_SELL = '#26A69A'      # 卖出信号 — 绿色
COLOR_EQUITY = '#1976D2'    # 净值曲线 — 蓝色
COLOR_BENCHMARK = '#FF9800' # 基准曲线 — 橙色
COLOR_DRAWDOWN = '#EF9A9A'  # 回撤区域 — 浅红色
COLOR_GRID = '#E0E0E0'      # 网格线


def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 的索引或 date 列为 datetime 类型。"""
    df = data.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if 'date' in df.columns:
        df = df.set_index('date')
    return df


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """保存图表或在屏幕上显示。"""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info("图表已保存至: %s", save_path)
        plt.close(fig)
    else:
        plt.show()


class ChartGenerator:
    """图表生成器 — 提供量化交易常用图表的静态方法集合。

    所有方法均为 @staticmethod，可直接调用::

        ChartGenerator.plot_candlestick(df, title="平安银行K线图")
    """

    # ================================================================ #
    #  K线图
    # ================================================================ #
    @staticmethod
    def plot_candlestick(
        data: pd.DataFrame,
        title: str = "K线图",
        save_path: Optional[str] = None,
    ) -> None:
        """绘制K线图（简化版 — 纯matplotlib实现）。

        上方: K线 (阳线红色, 阴线绿色 — 中国标准)
        下方: 成交量柱状图

        Parameters
        ----------
        data : pd.DataFrame
            至少包含 date/open/high/low/close/volume 列
        title : str
            图表标题
        save_path : str, optional
            图片保存路径，为 None 时直接显示
        """
        df = data.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("数据中缺少 'date' 列")

        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col not in df.columns:
                raise ValueError(f"数据中缺少 '{col}' 列")

        # 判断阳线/阴线
        up = df['close'] >= df['open']
        down = ~up

        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        x = np.arange(len(df))

        # ---------- K线主体 (实体) ----------
        body_width = 0.6
        # 阳线
        ax_price.bar(
            x[up], (df['close'] - df['open'])[up],
            bottom=df['open'][up],
            width=body_width, color=COLOR_UP, edgecolor=COLOR_UP, linewidth=0.5,
        )
        # 阴线
        ax_price.bar(
            x[down], (df['open'] - df['close'])[down],
            bottom=df['close'][down],
            width=body_width, color=COLOR_DOWN, edgecolor=COLOR_DOWN, linewidth=0.5,
        )

        # ---------- K线影线 (上下影线) ----------
        ax_price.vlines(x[up], df['low'][up], df['high'][up], color=COLOR_UP, linewidth=0.6)
        ax_price.vlines(x[down], df['low'][down], df['high'][down], color=COLOR_DOWN, linewidth=0.6)

        # ---------- 成交量柱状图 ----------
        vol_colors = [COLOR_UP if u else COLOR_DOWN for u in up]
        ax_vol.bar(x, df['volume'], width=body_width, color=vol_colors, alpha=0.7)
        ax_vol.set_ylabel('成交量', fontsize=10)

        # ---------- X轴日期标签 ----------
        # 自动选择刻度间隔
        n = len(df)
        if n <= 60:
            step = max(1, n // 10)
        elif n <= 250:
            step = max(1, n // 15)
        else:
            step = max(1, n // 20)

        tick_positions = list(range(0, n, step))
        tick_labels = [df['date'].iloc[i].strftime('%Y-%m-%d') for i in tick_positions]
        ax_vol.set_xticks(tick_positions)
        ax_vol.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

        # ---------- 美化 ----------
        ax_price.set_ylabel('价格 (元)', fontsize=10)
        ax_price.grid(True, alpha=0.3, color=COLOR_GRID)
        ax_vol.grid(True, alpha=0.3, color=COLOR_GRID)
        ax_price.set_xlim(-1, n)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save_or_show(fig, save_path)

    # ================================================================ #
    #  交易信号图
    # ================================================================ #
    @staticmethod
    def plot_signals(
        data: pd.DataFrame,
        title: str = "交易信号图",
        save_path: Optional[str] = None,
    ) -> None:
        """绘制交易信号图。

        - 收盘价折线图
        - 买入信号: 红色向上三角 ▲
        - 卖出信号: 绿色向下三角 ▼
        - 如果数据中存在 sma_* / ema_* 列，自动叠加均线

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 date, close 列。
            信号列: signal (值为 "buy"/"sell"/"hold") 或
                    buy_signal/sell_signal (布尔值)
        title : str
            图表标题
        save_path : str, optional
            图片保存路径
        """
        df = data.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("数据中缺少 'date' 列")
        if 'close' not in df.columns:
            raise ValueError("数据中缺少 'close' 列")

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        x = np.arange(len(df))

        # ---------- 收盘价折线 ----------
        ax.plot(x, df['close'], color='#424242', linewidth=1.2,
                label='收盘价', zorder=2)

        # ---------- 均线 (自动检测 sma_*/ema_* 列) ----------
        ma_colors = ['#FF9800', '#2196F3', '#9C27B0', '#F44336', '#4CAF50']
        ma_idx = 0
        for col in sorted(df.columns):
            if col.startswith(('sma_', 'ema_')):
                color = ma_colors[ma_idx % len(ma_colors)]
                label = col.upper().replace('_', ' ')
                ax.plot(x, df[col], color=color, linewidth=1.0,
                        linestyle='--', alpha=0.8, label=label, zorder=2)
                ma_idx += 1

        # ---------- 检测买卖信号 ----------
        buy_mask = pd.Series(False, index=df.index)
        sell_mask = pd.Series(False, index=df.index)

        if 'signal' in df.columns:
            buy_mask = df['signal'].astype(str).str.lower() == 'buy'
            sell_mask = df['signal'].astype(str).str.lower() == 'sell'
        if 'buy_signal' in df.columns:
            buy_mask = buy_mask | df['buy_signal'].astype(bool)
        if 'sell_signal' in df.columns:
            sell_mask = sell_mask | df['sell_signal'].astype(bool)

        # ---------- 绘制买卖点 ----------
        if buy_mask.any():
            ax.scatter(
                x[buy_mask], df['close'][buy_mask],
                marker='^', color=COLOR_BUY, s=120, edgecolors='white',
                linewidths=0.8, label='买入信号', zorder=5,
            )
        if sell_mask.any():
            ax.scatter(
                x[sell_mask], df['close'][sell_mask],
                marker='v', color=COLOR_SELL, s=120, edgecolors='white',
                linewidths=0.8, label='卖出信号', zorder=5,
            )

        # ---------- X轴日期标签 ----------
        n = len(df)
        step = max(1, n // 15)
        tick_positions = list(range(0, n, step))
        tick_labels = [df['date'].iloc[i].strftime('%Y-%m-%d') for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

        # ---------- 美化 ----------
        ax.set_ylabel('价格 (元)', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, color=COLOR_GRID)
        ax.set_xlim(-1, n)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save_or_show(fig, save_path)

    # ================================================================ #
    #  净值曲线图
    # ================================================================ #
    @staticmethod
    def plot_equity_curve(
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "策略净值曲线",
        save_path: Optional[str] = None,
    ) -> None:
        """绘制净值曲线图，含回撤区域填充。

        Parameters
        ----------
        equity_curve : pd.Series
            策略净值序列（索引为日期或整数）
        benchmark : pd.Series, optional
            基准净值序列（用于对比）
        title : str
            图表标题
        save_path : str, optional
            图片保存路径
        """
        fig, (ax_equity, ax_dd) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        equity = equity_curve.copy()
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)

        # ---------- 净值曲线 ----------
        ax_equity.plot(equity.index, equity.values,
                       color=COLOR_EQUITY, linewidth=1.5, label='策略净值')

        if benchmark is not None:
            bench = benchmark.copy()
            if not isinstance(bench.index, pd.DatetimeIndex):
                bench.index = pd.to_datetime(bench.index)
            ax_equity.plot(bench.index, bench.values,
                           color=COLOR_BENCHMARK, linewidth=1.2,
                           linestyle='--', alpha=0.8, label='基准净值')

        # ---------- 回撤计算与填充 ----------
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax  # 百分比回撤 (负值)

        # 在净值图上填充回撤区域
        ax_equity.fill_between(
            equity.index, equity.values, cummax.values,
            where=(equity < cummax),
            color=COLOR_DRAWDOWN, alpha=0.3, label='回撤区域',
        )

        # 下方回撤子图
        ax_dd.fill_between(
            drawdown.index, 0, drawdown.values,
            color=COLOR_DRAWDOWN, alpha=0.6,
        )
        ax_dd.plot(drawdown.index, drawdown.values,
                   color='#EF5350', linewidth=0.8)
        ax_dd.set_ylabel('回撤', fontsize=10)
        ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        # ---------- 美化 ----------
        ax_equity.set_ylabel('净值', fontsize=11)
        ax_equity.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax_equity.grid(True, alpha=0.3, color=COLOR_GRID)
        ax_dd.grid(True, alpha=0.3, color=COLOR_GRID)

        # X轴日期格式
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_dd.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=45)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save_or_show(fig, save_path)

    # ================================================================ #
    #  回撤图
    # ================================================================ #
    @staticmethod
    def plot_drawdown(
        drawdown_series: pd.Series,
        title: str = "策略回撤图",
        save_path: Optional[str] = None,
    ) -> None:
        """绘制回撤百分比随时间变化的图表。

        Parameters
        ----------
        drawdown_series : pd.Series
            回撤序列 (负值，如 -0.05 代表 5% 回撤)
        title : str
            图表标题
        save_path : str, optional
            图片保存路径
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        dd = drawdown_series.copy()
        if not isinstance(dd.index, pd.DatetimeIndex):
            dd.index = pd.to_datetime(dd.index)

        # 填充区域
        ax.fill_between(dd.index, 0, dd.values, color=COLOR_DRAWDOWN, alpha=0.6)
        ax.plot(dd.index, dd.values, color='#EF5350', linewidth=1.0)

        # 标注最大回撤点
        if len(dd) > 0:
            max_dd_idx = dd.idxmin()
            max_dd_val = dd.min()
            ax.annotate(
                f'最大回撤: {max_dd_val:.2%}',
                xy=(max_dd_idx, max_dd_val),
                xytext=(max_dd_idx, max_dd_val * 0.6),
                fontsize=10, color='#B71C1C', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5),
                ha='center',
            )

        # 美化
        ax.set_ylabel('回撤幅度', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.grid(True, alpha=0.3, color=COLOR_GRID)
        ax.set_ylim(top=0.01)  # 确保 0 线在上方

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=45)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save_or_show(fig, save_path)

    # ================================================================ #
    #  月度收益热力图
    # ================================================================ #
    @staticmethod
    def plot_monthly_returns(
        equity_curve: pd.Series,
        title: str = "月度收益热力图",
        save_path: Optional[str] = None,
    ) -> None:
        """绘制月度收益率热力图 (纯matplotlib实现)。

        行 = 年份，列 = 月份 (1~12)。
        单元格颜色: 红色正收益，绿色负收益，白色接近零。

        Parameters
        ----------
        equity_curve : pd.Series
            策略净值序列（索引为日期）
        title : str
            图表标题
        save_path : str, optional
            图片保存路径
        """
        equity = equity_curve.copy()
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)

        # 计算日收益率 → 按月汇总
        daily_returns = equity.pct_change().dropna()

        # 按 (年, 月) 分组计算月度收益率
        monthly = daily_returns.groupby(
            [daily_returns.index.year, daily_returns.index.month]
        ).apply(lambda x: (1 + x).prod() - 1)

        # 构建 年×月 矩阵
        if isinstance(monthly.index, pd.MultiIndex):
            years = sorted(monthly.index.get_level_values(0).unique())
            months = list(range(1, 13))
        else:
            logger.warning("月度收益数据不足，无法生成热力图")
            return

        matrix = pd.DataFrame(index=years, columns=months, dtype=float)
        for (year, month), value in monthly.items():
            matrix.loc[year, month] = value

        matrix = matrix.astype(float)

        # ---------- 绘制热力图 ----------
        fig, ax = plt.subplots(figsize=(14, max(3, len(years) * 0.8 + 1)))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 自定义红绿色图 (中国标准: 正收益红色, 负收益绿色)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#26A69A', '#E8F5E9', '#FFFFFF', '#FFEBEE', '#EF5350']
        cmap = LinearSegmentedColormap.from_list('cn_stock', colors_list, N=256)

        # 确定颜色范围 (对称)
        vmax = max(abs(matrix.min().min()), abs(matrix.max().max()), 0.01)
        if np.isnan(vmax):
            vmax = 0.1

        im = ax.imshow(
            matrix.values, cmap=cmap, aspect='auto',
            vmin=-vmax, vmax=vmax,
        )

        # 行列标签
        month_labels = [f'{m}月' for m in months]
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(month_labels, fontsize=10)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels([str(y) for y in years], fontsize=10)

        # 在每个格子中写入收益率数值
        for i in range(len(years)):
            for j in range(len(months)):
                val = matrix.iloc[i, j]
                if pd.notna(val):
                    text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:.1%}',
                            ha='center', va='center',
                            fontsize=8, fontweight='bold', color=text_color)
                else:
                    ax.text(j, i, '-',
                            ha='center', va='center',
                            fontsize=8, color='#BDBDBD')

        # 颜色条
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        cbar.set_label('月度收益率', fontsize=10)

        ax.set_xlabel('月份', fontsize=11)
        ax.set_ylabel('年份', fontsize=11)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save_or_show(fig, save_path)

    # ================================================================ #
    #  组合报告图 (便捷方法)
    # ================================================================ #
    @staticmethod
    def plot_backtest_report(
        data: pd.DataFrame,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "回测报告",
        save_path: Optional[str] = None,
    ) -> None:
        """生成完整回测报告图 (四合一面板)。

        包含: 交易信号图 / 净值曲线 / 回撤图 / 月度收益热力图

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV + 信号数据
        equity_curve : pd.Series
            策略净值序列
        benchmark : pd.Series, optional
            基准净值序列
        title : str
            图表标题
        save_path : str, optional
            图片保存路径
        """
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

        # 创建子图布局: 上 (信号图) / 中 (净值+回撤) / 下 (月度热力图)
        gs = fig.add_gridspec(4, 1, hspace=0.35, height_ratios=[3, 2, 1.5, 2])

        # ---- 子图1: 交易信号 ----
        ax1 = fig.add_subplot(gs[0])
        df = data.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        x = np.arange(len(df))
        ax1.plot(x, df['close'], color='#424242', linewidth=1.0, label='收盘价')

        # 均线
        ma_colors = ['#FF9800', '#2196F3', '#9C27B0']
        mi = 0
        for col in sorted(df.columns):
            if col.startswith(('sma_', 'ema_')):
                ax1.plot(x, df[col], color=ma_colors[mi % len(ma_colors)],
                         linewidth=0.8, linestyle='--', alpha=0.8,
                         label=col.upper().replace('_', ' '))
                mi += 1

        # 信号标注
        buy_mask = pd.Series(False, index=df.index)
        sell_mask = pd.Series(False, index=df.index)
        if 'signal' in df.columns:
            buy_mask = df['signal'].astype(str).str.lower() == 'buy'
            sell_mask = df['signal'].astype(str).str.lower() == 'sell'
        if 'buy_signal' in df.columns:
            buy_mask = buy_mask | df['buy_signal'].astype(bool)
        if 'sell_signal' in df.columns:
            sell_mask = sell_mask | df['sell_signal'].astype(bool)

        if buy_mask.any():
            ax1.scatter(x[buy_mask], df['close'][buy_mask],
                        marker='^', color=COLOR_BUY, s=80,
                        edgecolors='white', linewidths=0.5, label='买入', zorder=5)
        if sell_mask.any():
            ax1.scatter(x[sell_mask], df['close'][sell_mask],
                        marker='v', color=COLOR_SELL, s=80,
                        edgecolors='white', linewidths=0.5, label='卖出', zorder=5)

        ax1.set_title('交易信号', fontsize=12)
        ax1.legend(loc='best', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1, len(df))
        step = max(1, len(df) // 15)
        ticks = list(range(0, len(df), step))
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([df['date'].iloc[i].strftime('%m-%d') for i in ticks],
                            rotation=45, fontsize=7)

        # ---- 子图2: 净值曲线 ----
        ax2 = fig.add_subplot(gs[1])
        eq = equity_curve.copy()
        if not isinstance(eq.index, pd.DatetimeIndex):
            eq.index = pd.to_datetime(eq.index)

        ax2.plot(eq.index, eq.values, color=COLOR_EQUITY, linewidth=1.2, label='策略净值')
        if benchmark is not None:
            bench = benchmark.copy()
            if not isinstance(bench.index, pd.DatetimeIndex):
                bench.index = pd.to_datetime(bench.index)
            ax2.plot(bench.index, bench.values, color=COLOR_BENCHMARK,
                     linewidth=1.0, linestyle='--', alpha=0.8, label='基准')

        cummax = eq.cummax()
        ax2.fill_between(eq.index, eq.values, cummax.values,
                         where=(eq < cummax), color=COLOR_DRAWDOWN, alpha=0.3)

        ax2.set_title('净值曲线', fontsize=12)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # ---- 子图3: 回撤 ----
        ax3 = fig.add_subplot(gs[2], sharex=ax2)
        dd = (eq - cummax) / cummax
        ax3.fill_between(dd.index, 0, dd.values, color=COLOR_DRAWDOWN, alpha=0.6)
        ax3.plot(dd.index, dd.values, color='#EF5350', linewidth=0.8)
        ax3.set_title('回撤', fontsize=12)
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(top=0.01)

        # ---- 子图4: 月度收益热力图 ----
        ax4 = fig.add_subplot(gs[3])
        daily_ret = eq.pct_change().dropna()
        monthly = daily_ret.groupby(
            [daily_ret.index.year, daily_ret.index.month]
        ).apply(lambda g: (1 + g).prod() - 1)

        if isinstance(monthly.index, pd.MultiIndex):
            yrs = sorted(monthly.index.get_level_values(0).unique())
            mons = list(range(1, 13))
            mat = pd.DataFrame(index=yrs, columns=mons, dtype=float)
            for (y, m), v in monthly.items():
                mat.loc[y, m] = v
            mat = mat.astype(float)

            from matplotlib.colors import LinearSegmentedColormap
            cn_cmap = LinearSegmentedColormap.from_list(
                'cn', ['#26A69A', '#E8F5E9', '#FFFFFF', '#FFEBEE', '#EF5350'], N=256)
            vm = max(abs(mat.min().min()), abs(mat.max().max()), 0.01)
            if np.isnan(vm):
                vm = 0.1
            im = ax4.imshow(mat.values, cmap=cn_cmap, aspect='auto', vmin=-vm, vmax=vm)
            ax4.set_xticks(range(12))
            ax4.set_xticklabels([f'{m}月' for m in mons], fontsize=8)
            ax4.set_yticks(range(len(yrs)))
            ax4.set_yticklabels([str(y) for y in yrs], fontsize=9)
            for i in range(len(yrs)):
                for j in range(12):
                    v = mat.iloc[i, j]
                    if pd.notna(v):
                        tc = 'white' if abs(v) > vm * 0.6 else 'black'
                        ax4.text(j, i, f'{v:.1%}', ha='center', va='center',
                                 fontsize=7, fontweight='bold', color=tc)
            ax4.set_title('月度收益', fontsize=12)
            fig.colorbar(im, ax=ax4, shrink=0.6, pad=0.02)
        else:
            ax4.text(0.5, 0.5, '数据不足，无法生成月度热力图',
                     ha='center', va='center', fontsize=12, transform=ax4.transAxes)

        try:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception:
            pass  # 混合子图类型可能不兼容tight_layout，忽略即可
        _save_or_show(fig, save_path)
