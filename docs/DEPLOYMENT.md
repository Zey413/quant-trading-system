# 部署文档

本文档介绍量化交易系统的各种部署方式，包括 Docker 部署、Linux 服务器部署、定时任务配置和监控方案。

---

## 目录

- [Docker 部署](#docker-部署)
- [Docker Compose 部署](#docker-compose-部署)
- [Linux 服务器部署](#linux-服务器部署)
- [定时任务配置](#定时任务配置)
- [监控与告警](#监控与告警)
- [日志管理](#日志管理)
- [生产环境建议](#生产环境建议)

---

## Docker 部署

### 构建镜像

```bash
# 在项目根目录执行
docker build -t quant-trading:latest .
```

### 运行容器

```bash
# 基础运行
docker run --rm quant-trading:latest quant --help

# 获取数据
docker run --rm quant-trading:latest quant fetch -s 000001 --start 2024-01-01

# 运行回测
docker run --rm quant-trading:latest quant backtest -st ma_crossover -s 000001

# 挂载配置和数据目录
docker run --rm \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data_cache:/app/data_cache \
  -v $(pwd)/output:/app/output \
  quant-trading:latest \
  quant backtest -st ma_crossover -s 000001 --save /app/output/report.png
```

### 交互式使用

```bash
docker run -it \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data_cache:/app/data_cache \
  quant-trading:latest \
  /bin/bash
```

---

## Docker Compose 部署

### 启动服务

```bash
# 构建并启动
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f quant
```

### 执行命令

```bash
# 获取数据
docker-compose exec quant quant fetch -s 000001 --start 2024-01-01

# 运行回测
docker-compose exec quant quant backtest -st ma_crossover -s 000001

# 参数优化
docker-compose exec quant quant optimize -st ma_crossover -s 000001 \
  -p '{"short_window":[3,5,10],"long_window":[15,20,30]}'

# 生成报告
docker-compose exec quant quant report -st ma_crossover -s 000001 -f csv -o /app/output
```

### 停止服务

```bash
docker-compose down
```

### 自定义配置

在 `docker-compose.yml` 中可以修改：
- `volumes`: 挂载自定义配置文件和输出目录
- `environment`: 设置环境变量
- `command`: 设置默认启动命令

---

## Linux 服务器部署

### 前置条件

- Python 3.10+
- pip
- Git

### 安装步骤

```bash
# 1. 创建专用用户
sudo useradd -m -s /bin/bash quant
sudo su - quant

# 2. 克隆项目
git clone https://github.com/zey413/quant-trading-system.git
cd quant-trading-system

# 3. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 4. 安装
pip install -e .

# 5. 创建必要目录
mkdir -p data_cache output reports logs

# 6. 复制并编辑配置
cp config/default.yaml config/production.yaml
vim config/production.yaml

# 7. 验证
quant --help
quant fetch -s 000001 --start 2024-01-01 -c config/production.yaml
```

### Systemd 服务（可选）

如果需要作为后台服务运行（例如定时回测任务管理），可以创建 Systemd 服务文件：

```ini
# /etc/systemd/system/quant-backtest.service
[Unit]
Description=量化交易系统回测服务
After=network.target

[Service]
Type=oneshot
User=quant
WorkingDirectory=/home/quant/quant-trading-system
ExecStart=/home/quant/quant-trading-system/.venv/bin/quant backtest -st ma_crossover -s 000001 -c config/production.yaml
StandardOutput=append:/home/quant/quant-trading-system/logs/backtest.log
StandardError=append:/home/quant/quant-trading-system/logs/backtest_error.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl start quant-backtest
sudo systemctl status quant-backtest
```

---

## 定时任务配置

### Crontab 方式

```bash
# 编辑 crontab
crontab -e
```

常用定时任务示例：

```cron
# 每天收盘后(16:00)获取数据
0 16 * * 1-5 /home/quant/quant-trading-system/.venv/bin/quant fetch -s 000001 --start 2024-01-01 -c /home/quant/quant-trading-system/config/production.yaml >> /home/quant/quant-trading-system/logs/fetch.log 2>&1

# 每天17:00运行回测
0 17 * * 1-5 /home/quant/quant-trading-system/.venv/bin/quant backtest -st ma_crossover -s 000001 -c /home/quant/quant-trading-system/config/production.yaml >> /home/quant/quant-trading-system/logs/backtest.log 2>&1

# 每周一生成周报
0 18 * * 1 /home/quant/quant-trading-system/.venv/bin/quant report -st ma_crossover -s 000001 -f csv -o /home/quant/quant-trading-system/reports -c /home/quant/quant-trading-system/config/production.yaml >> /home/quant/quant-trading-system/logs/report.log 2>&1

# 每月1日运行参数优化
0 20 1 * * /home/quant/quant-trading-system/.venv/bin/quant optimize -st ma_crossover -s 000001 -p '{"short_window":[3,5,10],"long_window":[15,20,30]}' -c /home/quant/quant-trading-system/config/production.yaml >> /home/quant/quant-trading-system/logs/optimize.log 2>&1
```

### 批量回测脚本

创建批量回测脚本 `scripts/daily_backtest.sh`：

```bash
#!/bin/bash
# 每日批量回测脚本

set -e

QUANT="/home/quant/quant-trading-system/.venv/bin/quant"
CONFIG="/home/quant/quant-trading-system/config/production.yaml"
OUTPUT="/home/quant/quant-trading-system/output"
LOG="/home/quant/quant-trading-system/logs/daily_$(date +%Y%m%d).log"

echo "===== 开始每日回测 $(date) =====" >> "$LOG"

# 股票列表
STOCKS="000001 600519 000858 601318 000333"

# 策略列表
STRATEGIES="ma_crossover rsi macd bollinger"

for stock in $STOCKS; do
    for strategy in $STRATEGIES; do
        echo "回测: $stock - $strategy" >> "$LOG"
        $QUANT backtest -st "$strategy" -s "$stock" -c "$CONFIG" >> "$LOG" 2>&1 || true
        echo "---" >> "$LOG"
    done
done

echo "===== 每日回测完成 $(date) =====" >> "$LOG"
```

```bash
chmod +x scripts/daily_backtest.sh
```

---

## 监控与告警

### 日志监控

系统使用 Python 标准 logging 模块。通过 `--verbose` / `-v` 开启详细日志。

在 Python 代码中配置日志：

```python
import logging

# 详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="logs/quant.log",
)

# 仅警告
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="logs/quant.log",
)
```

### 简易告警脚本

创建 `scripts/alert.sh` 监控回测结果：

```bash
#!/bin/bash
# 简易告警：检查回测日志中的异常

LOG_DIR="/home/quant/quant-trading-system/logs"
ALERT_LOG="$LOG_DIR/alerts.log"

# 检查最近的日志文件
LATEST_LOG=$(ls -t "$LOG_DIR"/backtest*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "$(date): 未找到回测日志" >> "$ALERT_LOG"
    exit 0
fi

# 检查是否有错误
if grep -q "错误\|Error\|FAILED\|失败" "$LATEST_LOG"; then
    echo "$(date): 发现回测错误！" >> "$ALERT_LOG"
    grep "错误\|Error\|FAILED\|失败" "$LATEST_LOG" >> "$ALERT_LOG"
    # 可选：发送邮件或其他通知
    # mail -s "量化回测告警" admin@example.com < "$ALERT_LOG"
fi
```

### 磁盘空间监控

```bash
#!/bin/bash
# 检查缓存目录大小
CACHE_SIZE=$(du -sh /home/quant/quant-trading-system/data_cache 2>/dev/null | cut -f1)
echo "数据缓存大小: $CACHE_SIZE"

# 清理超过30天的旧日志
find /home/quant/quant-trading-system/logs -name "*.log" -mtime +30 -delete

# 清理超过90天的旧报告
find /home/quant/quant-trading-system/reports -mtime +90 -delete
```

---

## 日志管理

### 日志轮转

使用 logrotate 管理日志文件：

```conf
# /etc/logrotate.d/quant-trading
/home/quant/quant-trading-system/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 quant quant
}
```

### 日志级别说明

| 级别 | 适用场景 |
|:-----|:---------|
| `DEBUG` | 开发调试，输出详细信息 |
| `INFO` | 正常运行，关键操作记录 |
| `WARNING` | 潜在问题（默认） |
| `ERROR` | 运行错误 |

---

## 生产环境建议

### 安全建议

1. **配置文件权限**: Tushare Token 等敏感信息不要提交到 Git

```bash
chmod 600 config/production.yaml
echo "config/production.yaml" >> .gitignore
```

2. **使用环境变量**: 敏感信息通过环境变量注入

```bash
export TUSHARE_TOKEN="your_token"
```

3. **专用用户**: 使用非 root 用户运行

### 性能建议

1. **启用数据缓存**: 避免重复网络请求

```yaml
data_source:
  cache_enabled: true
  cache_dir: "data_cache"
```

2. **限制并发**: 参数优化时控制参数空间大小

3. **磁盘空间**: 定期清理旧的缓存数据和日志

### 备份建议

```bash
# 备份配置和输出数据
tar -czf quant_backup_$(date +%Y%m%d).tar.gz \
  config/ output/ reports/ data_cache/

# 定期备份到远程
rsync -avz config/ reports/ user@backup-server:/backup/quant/
```

### 更新部署

```bash
cd /home/quant/quant-trading-system

# 拉取最新代码
git pull origin main

# 更新依赖
source .venv/bin/activate
pip install -e .

# 验证
quant --help
pytest tests/ -v
```
