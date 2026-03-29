"""应用配置 - 使用Pydantic进行配置管理"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


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


class BacktestConfig(BaseModel):
    """回测配置"""
    initial_capital: float = Field(default=1_000_000.0, description="初始资金(元)")
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    benchmark: str = "000300"  # 沪深300
    risk_free_rate: float = Field(default=0.025, description="无风险利率(10年期国债)")


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
