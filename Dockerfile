# ============================================================
# 量化交易系统 Dockerfile
# 基于 Python 3.11 slim，最小化镜像体积
# ============================================================

FROM python:3.11-slim AS base

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 系统依赖（matplotlib 需要字体渲染相关库）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libfreetype6-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r quant && useradd -r -g quant -m quant

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY pyproject.toml ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -e . 2>/dev/null || \
    pip install --no-cache-dir \
        "pandas>=2.0.0" \
        "numpy>=1.24.0" \
        "matplotlib>=3.7.0" \
        "pydantic>=2.0.0" \
        "pyyaml>=6.0" \
        "click>=8.1.0" \
        "rich>=13.0.0" \
        "pyarrow>=12.0.0" \
        "akshare>=1.10.0" \
        "tushare>=1.4.0"

# 复制项目源码
COPY . .

# 安装项目
RUN pip install --no-cache-dir -e .

# 创建数据和输出目录
RUN mkdir -p data_cache output reports logs && \
    chown -R quant:quant /app

# 切换到非 root 用户
USER quant

# 默认配置文件
ENV QUANT_CONFIG=/app/config/default.yaml

# 健康检查
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD quant --help > /dev/null 2>&1 || exit 1

# 默认入口
ENTRYPOINT ["quant"]
CMD ["--help"]
