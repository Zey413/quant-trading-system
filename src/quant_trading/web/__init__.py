"""Web Dashboard 模块 - 基于 Streamlit 的量化交易可视化平台

启动方式::

    streamlit run src/quant_trading/web/app.py

或者通过辅助函数::

    from quant_trading.web import launch
    launch()
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch(port: int = 8501, **kwargs) -> None:
    """便捷启动 Streamlit Dashboard

    Parameters
    ----------
    port : int
        监听端口，默认 8501
    **kwargs
        传递给 ``streamlit run`` 的额外参数
    """
    app_path = Path(__file__).parent / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    for k, v in kwargs.items():
        cmd.extend([f"--{k}", str(v)])
    subprocess.run(cmd, check=True)


__all__ = ["launch"]
