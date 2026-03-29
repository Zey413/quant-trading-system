"""回测报告与数据导出测试

测试覆盖：
- ReportGenerator: 文本报告生成、CSV交易导出、净值曲线导出
- DataExporter: CSV / Excel / JSON 导出
- 边界情况: 空数据、无交易
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest.metrics import PerformanceMetrics, PerformanceResult
from quant_trading.backtest.report import ReportGenerator
from quant_trading.core.config import AppConfig, BacktestConfig
from quant_trading.core.models import TradeRecord
from quant_trading.data.export import DataExporter


# ============================================================
# Fixtures & Helpers
# ============================================================


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        backtest=BacktestConfig(
            initial_capital=1_000_000.0,
            start_date="2024-01-01",
            end_date="2024-12-31",
            risk_free_rate=0.025,
        ),
    )


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """构造一条有涨有跌的净值曲线"""
    n_days = 60
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.015, n_days - 1)
    values = [1_000_000.0]
    for r in daily_returns:
        values.append(values[-1] * (1 + r))
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_trades() -> list[TradeRecord]:
    """构造一组包含盈亏的交易记录"""
    return [
        TradeRecord("t1", date(2024, 1, 10), "000001", "sell", 1000, 11.0, 5.0, 11.0, pnl=950.0),
        TradeRecord("t2", date(2024, 2, 5), "000001", "sell", 1000, 9.5, 5.0, 9.5, pnl=-520.0),
        TradeRecord("t3", date(2024, 3, 15), "000001", "sell", 500, 12.0, 5.0, 6.0, pnl=480.0),
        TradeRecord("t4", date(2024, 4, 20), "000001", "sell", 800, 8.5, 5.0, 6.8, pnl=-1200.0),
        TradeRecord("t5", date(2024, 5, 1), "000001", "sell", 600, 13.0, 5.0, 7.8, pnl=1800.0),
        TradeRecord("t6", date(2024, 6, 10), "000001", "sell", 1000, 10.0, 5.0, 10.0, pnl=-50.0),
    ]


@pytest.fixture
def sample_result(sample_equity_curve: pd.Series, sample_trades: list[TradeRecord]) -> PerformanceResult:
    """基于样本数据构造 PerformanceResult"""
    config = BacktestConfig(initial_capital=1_000_000.0, risk_free_rate=0.025)
    return PerformanceMetrics.calculate(sample_equity_curve, sample_trades, config)


def _make_empty_result() -> PerformanceResult:
    """构造空的绩效结果"""
    return PerformanceMetrics.calculate(
        pd.Series(dtype=float),
        [],
        BacktestConfig(initial_capital=1_000_000.0),
    )


# ============================================================
# ReportGenerator - 文本报告测试
# ============================================================


class TestReportGeneratorText:
    """文本报告生成测试"""

    def test_report_contains_header(
        self, sample_result: PerformanceResult
    ) -> None:
        """测试报告包含头部信息"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
            strategy_name="ma_crossover",
            symbol="000001",
        )
        assert "回测分析报告" in report
        assert "ma_crossover" in report
        assert "000001" in report

    def test_report_contains_metrics(
        self, sample_result: PerformanceResult
    ) -> None:
        """测试报告包含绩效指标"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
        )
        assert "绩效指标" in report
        assert "总收益率" in report
        assert "年化收益率" in report
        assert "最大回撤" in report
        assert "夏普比率" in report
        assert "Sortino比率" in report
        assert "Calmar比率" in report

    def test_report_contains_trade_stats(
        self, sample_result: PerformanceResult
    ) -> None:
        """测试报告包含交易统计"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
        )
        assert "交易统计" in report
        assert "总交易次数" in report
        assert "胜率" in report
        assert "盈亏比" in report

    def test_report_with_config_shows_capital(
        self,
        sample_result: PerformanceResult,
        app_config: AppConfig,
    ) -> None:
        """测试报告含资金信息"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
            config=app_config,
        )
        assert "资金概况" in report
        assert "初始资金" in report
        assert "期末资产" in report
        assert "净利润" in report

    def test_report_with_trades_shows_top5(
        self,
        sample_result: PerformanceResult,
        sample_trades: list[TradeRecord],
    ) -> None:
        """测试报告包含 Top 5 盈利/亏损交易"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
            trades=sample_trades,
        )
        assert "Top 5 盈利" in report
        assert "Top 5 亏损" in report

    def test_report_contains_monthly_returns(
        self, sample_result: PerformanceResult
    ) -> None:
        """测试报告包含月度收益率"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
        )
        assert "月度收益率" in report

    def test_report_contains_risk_analysis(
        self, sample_result: PerformanceResult
    ) -> None:
        """测试报告包含风险分析"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
        )
        assert "风险分析" in report

    def test_report_with_empty_result(self) -> None:
        """测试空结果的报告生成"""
        empty_result = _make_empty_result()
        report = ReportGenerator.generate_text_report(
            result=empty_result,
            strategy_name="test",
            symbol="000001",
        )
        # 应能生成报告而不报错
        assert "回测分析报告" in report
        assert "总收益率" in report
        # 空结果不应有月度收益和风险分析
        assert "报告结束" in report

    def test_report_date_range_from_equity(
        self, sample_result: PerformanceResult
    ) -> None:
        """测试从净值曲线推断日期范围"""
        report = ReportGenerator.generate_text_report(
            result=sample_result,
        )
        assert "回测区间" in report
        assert "2024" in report


# ============================================================
# ReportGenerator - CSV 导出测试
# ============================================================


class TestReportGeneratorCSV:
    """CSV 导出测试"""

    def test_csv_trades_export(
        self, tmp_path: Path, sample_trades: list[TradeRecord]
    ) -> None:
        """测试交易记录 CSV 导出"""
        filepath = str(tmp_path / "trades.csv")
        ReportGenerator.generate_csv_trades(sample_trades, filepath)

        assert Path(filepath).exists()

        # 读取并验证
        with open(filepath, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 头部 + 6 条记录
        assert len(rows) == 7
        header = rows[0]
        assert "trade_id" in header
        assert "pnl" in header
        assert "symbol" in header

    def test_csv_trades_empty(self, tmp_path: Path) -> None:
        """测试空交易记录导出"""
        filepath = str(tmp_path / "empty_trades.csv")
        ReportGenerator.generate_csv_trades([], filepath)

        assert Path(filepath).exists()

        with open(filepath, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 只有头部行
        assert len(rows) == 1

    def test_csv_trades_content_values(
        self, tmp_path: Path, sample_trades: list[TradeRecord]
    ) -> None:
        """测试 CSV 内容中数值正确"""
        filepath = str(tmp_path / "trades_values.csv")
        ReportGenerator.generate_csv_trades(sample_trades, filepath)

        df = pd.read_csv(filepath, dtype={"symbol": str})
        assert len(df) == 6
        # 第一条交易 pnl = 950.0
        assert abs(df.iloc[0]["pnl"] - 950.0) < 0.01
        assert df.iloc[0]["symbol"] == "000001"
        assert df.iloc[0]["side"] == "sell"

    def test_csv_trades_creates_parent_dirs(self, tmp_path: Path) -> None:
        """测试自动创建父目录"""
        filepath = str(tmp_path / "deep" / "nested" / "trades.csv")
        ReportGenerator.generate_csv_trades([], filepath)
        assert Path(filepath).exists()


# ============================================================
# ReportGenerator - 净值曲线导出测试
# ============================================================


class TestReportGeneratorEquity:
    """净值曲线 CSV 导出测试"""

    def test_equity_csv_export(
        self, tmp_path: Path, sample_equity_curve: pd.Series
    ) -> None:
        """测试净值曲线 CSV 导出"""
        filepath = str(tmp_path / "equity.csv")
        ReportGenerator.generate_equity_csv(sample_equity_curve, filepath)

        assert Path(filepath).exists()

        df = pd.read_csv(filepath)
        assert "date" in df.columns
        assert "equity" in df.columns
        assert "daily_return" in df.columns
        assert "cumulative_return" in df.columns
        assert len(df) == len(sample_equity_curve)

    def test_equity_csv_first_cumulative_return_zero(
        self, tmp_path: Path, sample_equity_curve: pd.Series
    ) -> None:
        """测试首日累计收益率为 0"""
        filepath = str(tmp_path / "equity_check.csv")
        ReportGenerator.generate_equity_csv(sample_equity_curve, filepath)

        df = pd.read_csv(filepath)
        assert abs(df.iloc[0]["cumulative_return"]) < 1e-10

    def test_equity_csv_empty(self, tmp_path: Path) -> None:
        """测试空净值曲线导出"""
        filepath = str(tmp_path / "empty_equity.csv")
        empty_equity = pd.Series(dtype=float)
        ReportGenerator.generate_equity_csv(empty_equity, filepath)

        assert Path(filepath).exists()
        df = pd.read_csv(filepath)
        assert len(df) == 0


# ============================================================
# DataExporter Tests
# ============================================================


class TestDataExporter:
    """数据导出工具测试"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "open": [10.0, 10.5, 10.3, 10.8, 11.0],
            "close": [10.5, 10.3, 10.8, 11.0, 10.9],
            "volume": [1000, 1200, 1100, 1300, 900],
        })

    def test_to_csv(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """测试 CSV 导出"""
        filepath = str(tmp_path / "test.csv")
        DataExporter.to_csv(sample_df, filepath)

        assert Path(filepath).exists()
        df = pd.read_csv(filepath)
        assert len(df) == 5
        assert "open" in df.columns
        assert "close" in df.columns

    def test_to_json(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """测试 JSON 导出"""
        filepath = str(tmp_path / "test.json")
        DataExporter.to_json(sample_df, filepath)

        assert Path(filepath).exists()
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 5
        assert "open" in data[0]
        assert "close" in data[0]

    def test_to_csv_creates_dirs(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """测试自动创建父目录"""
        filepath = str(tmp_path / "sub" / "dir" / "data.csv")
        DataExporter.to_csv(sample_df, filepath)
        assert Path(filepath).exists()

    def test_to_json_creates_dirs(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """测试 JSON 导出自动创建目录"""
        filepath = str(tmp_path / "sub" / "dir" / "data.json")
        DataExporter.to_json(sample_df, filepath)
        assert Path(filepath).exists()

    def test_to_csv_empty_df(self, tmp_path: Path) -> None:
        """测试空 DataFrame CSV 导出"""
        filepath = str(tmp_path / "empty.csv")
        DataExporter.to_csv(pd.DataFrame(), filepath)
        assert Path(filepath).exists()
        # 空 DataFrame 仍然可以被读取（可能仅含空内容）
        content = Path(filepath).read_text(encoding="utf-8-sig")
        # 文件存在但内容为空或仅换行
        assert content.strip() == ""

    def test_to_json_empty_df(self, tmp_path: Path) -> None:
        """测试空 DataFrame JSON 导出"""
        filepath = str(tmp_path / "empty.json")
        DataExporter.to_json(pd.DataFrame(), filepath)
        assert Path(filepath).exists()
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        assert data == []

    def test_to_excel_missing_openpyxl(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """测试 Excel 导出 - openpyxl 可能已安装也可能未安装"""
        filepath = str(tmp_path / "test.xlsx")
        try:
            DataExporter.to_excel(sample_df, filepath)
            # 如果成功说明 openpyxl 已安装
            assert Path(filepath).exists()
        except ImportError as e:
            assert "openpyxl" in str(e)


# ============================================================
# 辅助方法测试
# ============================================================


class TestReportHelpers:
    """报告辅助方法测试"""

    def test_monthly_returns_table_non_empty(
        self, sample_equity_curve: pd.Series
    ) -> None:
        """测试月度收益表非空"""
        table = ReportGenerator._build_monthly_returns_table(
            sample_equity_curve
        )
        assert len(table) > 0
        # 应包含年份标题行
        assert any("2024" in line for line in table)

    def test_monthly_returns_table_empty_data(self) -> None:
        """测试空数据的月度收益表"""
        table = ReportGenerator._build_monthly_returns_table(
            pd.Series(dtype=float)
        )
        assert table == []

    def test_monthly_returns_table_single_point(self) -> None:
        """测试单个数据点"""
        eq = pd.Series(
            [1_000_000.0],
            index=pd.bdate_range("2024-01-01", periods=1),
        )
        table = ReportGenerator._build_monthly_returns_table(eq)
        assert table == []

    def test_find_drawdown_periods(
        self, sample_result: PerformanceResult
    ) -> None:
        """测试回撤期间识别"""
        periods = ReportGenerator._find_drawdown_periods(
            sample_result.drawdown_series
        )
        # 有涨有跌的曲线应该有回撤期间
        assert len(periods) > 0
        # 按最大回撤降序排列
        if len(periods) > 1:
            assert periods[0]["max_drawdown"] >= periods[1]["max_drawdown"]
        # 每个期间包含必要字段
        for p in periods:
            assert "start" in p
            assert "end" in p
            assert "duration" in p
            assert "max_drawdown" in p
            assert p["duration"] > 0
            assert p["max_drawdown"] > 0

    def test_find_drawdown_periods_empty(self) -> None:
        """测试空回撤序列"""
        periods = ReportGenerator._find_drawdown_periods(
            pd.Series(dtype=float)
        )
        assert periods == []

    def test_find_drawdown_periods_no_drawdown(self) -> None:
        """测试无回撤的序列（纯涨）"""
        # 全0回撤
        dd = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            index=pd.bdate_range("2024-01-01", periods=5),
        )
        periods = ReportGenerator._find_drawdown_periods(dd)
        assert periods == []
