"""应用配置 - 使用Pydantic进行配置管理

支持 YAML 文件加载/保存，以及全面的字段与跨字段校验。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# 合法日期格式: YYYY-MM-DD 或 YYYYMMDD
_DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}$|^\d{8}$"
)


def _validate_date_str(value: str, field_name: str) -> str:
    """校验日期字符串格式与合法性

    接受 YYYY-MM-DD 或 YYYYMMDD，并验证日期本身合法（如不接受2月30日）。
    """
    if not _DATE_PATTERN.match(value):
        raise ValueError(
            f"{field_name} 格式错误: {value!r}，"
            "必须为 YYYY-MM-DD 或 YYYYMMDD"
        )
    # 进一步验证日期合法性
    fmt = "%Y-%m-%d" if "-" in value else "%Y%m%d"
    try:
        datetime.strptime(value, fmt)
    except ValueError:
        raise ValueError(
            f"{field_name} 日期无效: {value!r}"
        )
    return value


class DataSourceConfig(BaseModel):
    """数据源配置"""
    default_source: str = "akshare"
    tushare_token: str = ""
    cache_dir: str = "data_cache"
    cache_enabled: bool = True


class BrokerConfig(BaseModel):
    """券商/交易配置 - A股规则"""
    commission_rate: float = Field(default=0.0003, description="佣金率(万三)")
    min_commission: float = Field(default=5.0, description="最低佣金(元)")
    stamp_tax_rate: float = Field(default=0.001, description="印花税率(千一,仅卖出)")
    slippage: float = Field(default=0.001, description="滑点(千一)")
    lot_size: int = Field(default=100, description="最小交易单位(手)")
    trading_days_per_year: int = Field(default=244, description="年交易日数")
    t_plus_1: bool = Field(default=True, description="T+1规则")

    @field_validator("commission_rate")
    @classmethod
    def _check_commission_rate(cls, v: float) -> float:
        if not (0 <= v <= 0.01):
            raise ValueError(
                f"commission_rate 必须在 0 ~ 0.01 之间，当前值: {v}"
            )
        return v


class BacktestConfig(BaseModel):
    """回测配置"""
    initial_capital: float = Field(default=1_000_000.0, description="初始资金(元)")
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    benchmark: str = "000300"  # 沪深300
    risk_free_rate: float = Field(default=0.025, description="无风险利率(10年期国债)")

    @field_validator("initial_capital")
    @classmethod
    def _check_initial_capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"initial_capital 必须大于 0，当前值: {v}"
            )
        return v

    @field_validator("start_date")
    @classmethod
    def _check_start_date(cls, v: str) -> str:
        return _validate_date_str(v, "start_date")

    @field_validator("end_date")
    @classmethod
    def _check_end_date(cls, v: str) -> str:
        return _validate_date_str(v, "end_date")

    @model_validator(mode="after")
    def _check_date_range(self) -> BacktestConfig:
        """校验 start_date 在 end_date 之前"""
        fmt_start = "%Y-%m-%d" if "-" in self.start_date else "%Y%m%d"
        fmt_end = "%Y-%m-%d" if "-" in self.end_date else "%Y%m%d"
        start = datetime.strptime(self.start_date, fmt_start)
        end = datetime.strptime(self.end_date, fmt_end)
        if start >= end:
            raise ValueError(
                f"start_date ({self.start_date}) 必须早于 end_date ({self.end_date})"
            )
        return self


class RiskConfig(BaseModel):
    """风险管理配置"""
    max_position_pct: float = Field(default=0.3, description="单只股票最大持仓比例")
    max_total_position_pct: float = Field(default=0.8, description="最大总持仓比例")
    stop_loss_pct: float = Field(default=0.05, description="止损百分比")
    take_profit_pct: float = Field(default=0.15, description="止盈百分比")
    position_sizing_method: str = "fixed_ratio"
    fixed_ratio: float = Field(default=0.1, description="固定比例仓位")


class AppConfig(BaseModel):
    """应用全局配置"""
    data_source: DataSourceConfig = DataSourceConfig()
    broker: BrokerConfig = BrokerConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk: RiskConfig = RiskConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """从YAML文件加载配置"""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """保存配置到YAML文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, allow_unicode=True, default_flow_style=False)

    def validate(self) -> list[str]:
        """手动校验所有约束，返回错误信息列表（空列表表示全部通过）。

        该方法聚合 Pydantic 字段验证器与额外的跨字段逻辑，
        方便在 CLI / 脚本中一次性展示所有配置问题。

        Returns:
            list[str]: 校验错误描述。空列表表示配置完全合法。
        """
        errors: list[str] = []

        # --- BrokerConfig ---
        if not (0 <= self.broker.commission_rate <= 0.01):
            errors.append(
                f"broker.commission_rate 必须在 0~0.01 之间，"
                f"当前值: {self.broker.commission_rate}"
            )

        # --- BacktestConfig ---
        if self.backtest.initial_capital <= 0:
            errors.append(
                f"backtest.initial_capital 必须大于 0，"
                f"当前值: {self.backtest.initial_capital}"
            )

        for field_name in ("start_date", "end_date"):
            value = getattr(self.backtest, field_name)
            if not _DATE_PATTERN.match(value):
                errors.append(
                    f"backtest.{field_name} 格式错误: {value!r}，"
                    "必须为 YYYY-MM-DD 或 YYYYMMDD"
                )
            else:
                fmt = "%Y-%m-%d" if "-" in value else "%Y%m%d"
                try:
                    datetime.strptime(value, fmt)
                except ValueError:
                    errors.append(f"backtest.{field_name} 日期无效: {value!r}")

        # 跨字段: start_date < end_date
        try:
            fmt_s = "%Y-%m-%d" if "-" in self.backtest.start_date else "%Y%m%d"
            fmt_e = "%Y-%m-%d" if "-" in self.backtest.end_date else "%Y%m%d"
            sd = datetime.strptime(self.backtest.start_date, fmt_s)
            ed = datetime.strptime(self.backtest.end_date, fmt_e)
            if sd >= ed:
                errors.append(
                    f"backtest.start_date ({self.backtest.start_date}) "
                    f"必须早于 end_date ({self.backtest.end_date})"
                )
        except ValueError:
            pass  # 日期格式错误已在上面记录

        return errors
