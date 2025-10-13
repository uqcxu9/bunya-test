#!/bin/bash

echo "=========================================="
echo "  实验恢复任务 - 完整检查流程"
echo "=========================================="
echo ""

cd ~/projects/ACL24-EconAgent/ACL24-EconAgent-master

# ========== 1. 网络检查 ==========
echo "=== 1. 网络连接检查 ==="
echo "检查互联网连接..."
if ping -c 3 8.8.8.8 > /dev/null 2>&1; then
    echo "✅ 互联网连接正常"
else
    echo "❌ 警告: 无法连接到互联网"
fi

echo ""
echo "检查 DNS 解析..."
if nslookup google.com > /dev/null 2>&1; then
    echo "✅ DNS 解析正常"
else
    echo "❌ 警告: DNS 解析失败"
fi

echo ""
echo "检查活动网络连接..."
netstat -tnp 2>/dev/null | grep ESTABLISHED | head -5
echo ""

# ========== 2. 端口检查 ==========
echo "=== 2. 端口和防火墙检查 ==="
echo "检查常用端口状态..."

# 检查 HTTPS (443) 是否可达
if timeout 3 bash -c "echo > /dev/tcp/8.8.8.8/443" 2>/dev/null; then
    echo "✅ 端口 443 (HTTPS) 可达"
else
    echo "⚠️  端口 443 可能被阻塞"
fi

# 检查是否有僵尸连接
ZOMBIE_CONN=$(netstat -tn 2>/dev/null | grep CLOSE_WAIT | wc -l)
if [ $ZOMBIE_CONN -gt 0 ]; then
    echo "⚠️  发现 $ZOMBIE_CONN 个 CLOSE_WAIT 连接"
    echo "   建议重启网络或等待连接超时"
else
    echo "✅ 没有僵尸连接"
fi

echo ""

# ========== 3. API 连接测试 ==========
echo "=== 3. LLM API 连接测试 ==="
echo "测试 API 端点是否可达..."

# 这里假设你使用的是阿里云的 API，根据实际情况修改
API_HOST="dashscope.aliyuncs.com"
if timeout 5 bash -c "echo > /dev/tcp/$API_HOST/443" 2>/dev/null; then
    echo "✅ API 服务器 ($API_HOST) 可达"
else
    echo "⚠️  无法连接到 API 服务器 ($API_HOST)"
    echo "   这可能导致任务失败"
fi

echo ""

# ========== 4. 文件系统检查 ==========
echo "=== 4. 文件系统和资源检查 ==="

# 检查磁盘空间
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')
echo "可用磁盘空间: $DISK_AVAIL"

DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "❌ 警告: 磁盘使用率过高 ($DISK_USAGE%)"
else
    echo "✅ 磁盘空间充足"
fi

# 检查内存
echo ""
echo "内存使用情况:"
free -h | grep -E "Mem|Swap"

MEM_AVAIL=$(free -g | grep Mem | awk '{print $7}')
if [ $MEM_AVAIL -lt 5 ]; then
    echo "⚠️  警告: 可用内存较低 (${MEM_AVAIL}GB)"
else
    echo "✅ 内存充足 (${MEM_AVAIL}GB 可用)"
fi

echo ""

# ========== 5. 检查点文件验证 ==========
echo "=== 5. 检查点文件验证 ==="

CHECKPOINT="./test_run/econagent-paper-replication/data/gpt-3-perception-reflection-1-100agents-240months/env_156.pkl"

if [ -f "$CHECKPOINT" ]; then
    echo "✅ 检查点文件存在"
    ls -lh "$CHECKPOINT"
    
    # 检查其他必需文件
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
    if [ -f "$CHECKPOINT_DIR/obs_156.pkl" ] && \
       [ -f "$CHECKPOINT_DIR/dialog_156.pkl" ] && \
       [ -f "$CHECKPOINT_DIR/dialog4ref_156.pkl" ]; then
        echo "✅ 所有必需的检查点文件都存在"
    else
        echo "❌ 错误: 缺少必需的检查点文件"
        exit 1
    fi
else
    echo "❌ 错误: 检查点文件不存在: $CHECKPOINT"
    exit 1
fi

echo ""

# ========== 6. 检查 Python 环境 ==========
echo "=== 6. Python 环境检查 ==="

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✅ Python 版本: $PYTHON_VERSION"
else
    echo "❌ 错误: 找不到 Python"
    exit 1
fi

# 检查关键依赖
echo "检查关键 Python 包..."
python -c "import ai_economist, numpy, yaml, pickle" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python 依赖包正常"
else
    echo "❌ 错误: Python 依赖包缺失"
    exit 1
fi

echo ""

# ========== 7. 检查 simulate_resume.py ==========
echo "=== 7. 检查恢复脚本 ==="

if [ -f "simulate_resume.py" ]; then
    echo "✅ simulate_resume.py 存在"
    
    # 检查是否包含恢复逻辑
    if grep -q "resume_checkpoint" simulate_resume.py; then
        echo "✅ 恢复逻辑已添加"
    else
        echo "❌ 错误: simulate_resume.py 缺少恢复逻辑"
        echo "   请确保已添加 resume_checkpoint 参数"
        exit 1
    fi
else
    echo "❌ 错误: simulate_resume.py 不存在"
    echo "   正在从 simulate.py 创建..."
    cp simulate.py simulate_resume.py
    echo "⚠️  请手动编辑 simulate_resume.py 添加恢复功能"
    exit 1
fi

echo ""

# ========== 8. 备份现有数据 ==========
echo "=== 8. 备份现有数据 ==="

BACKUP_FILE="backup_step156_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "创建备份: $BACKUP_FILE"

tar -czf "$BACKUP_FILE" \
    test_run/econagent-paper-replication/data/gpt-3-perception-reflection-1-100agents-240months/ \
    2>/dev/null

if [ $? -eq 0 ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "✅ 备份完成 (大小: $BACKUP_SIZE)"
else
    echo "❌ 备份失败"
    exit 1
fi

echo ""

# ========== 9. 创建日志目录 ==========
echo "=== 9. 准备日志目录 ==="
mkdir -p logs
echo "✅ 日志目录已准备: ./logs/"

echo ""

# ========== 10. 总结和确认 ==========
echo "=========================================="
echo "  检查完成 - 准备启动任务"
echo "=========================================="
echo ""
echo "任务配置:"
echo "  - 起始步数: 156"
echo "  - 目标步数: 240"
echo "  - 剩余步数: 84 (35%)"
echo "  - 模型: qwen2.5-72b-instruct"
echo "  - 智能体数: 100"
echo ""
echo "预计时间: 约 1-1.5 天"
echo ""

read -p "是否继续启动任务? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "任务已取消"
    exit 0
fi

# ========== 11. 在 tmux 中启动 ==========
echo ""
echo "=== 在 tmux 中启动任务 ==="

# 检查 tmux 是否安装
if ! command -v tmux &> /dev/null; then
    echo "❌ 错误: tmux 未安装"
    echo "   安装命令: sudo apt-get install tmux"
    exit 1
fi

# 创建或连接到 tmux session
SESSION_NAME="econagent_resume"

# 如果 session 已存在，询问是否杀掉
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  发现已存在的 tmux session: $SESSION_NAME"
    read -p "是否杀掉旧 session? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "✅ 已杀掉旧 session"
    else
        echo "请手动处理旧 session: tmux attach -t $SESSION_NAME"
        exit 1
    fi
fi

# 创建新的 tmux session 并运行任务
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"

# 在 tmux 中运行命令
tmux send-keys -t "$SESSION_NAME" "python simulate_resume.py \
    --policy_model gpt \
    --num_agents 100 \
    --episode_length 240 \
    --dialog_len 3 \
    --model_name 'qwen2.5-72b-instruct' \
    --save_dir 'test_run/econagent-paper-replication' \
    --resume_checkpoint './test_run/econagent-paper-replication/data/gpt-3-perception-reflection-1-100agents-240months/env_156.pkl' \
    2>&1 | tee logs/econagent_resume_156.log" C-m

echo ""
echo "=========================================="
echo "  ✅ 任务已在 tmux 中启动!"
echo "=========================================="
echo ""
echo "查看和管理:"
echo "  - 连接到 session:   tmux attach -t $SESSION_NAME"
echo "  - 断开连接:         Ctrl+B 然后按 D"
echo "  - 查看日志:         tail -f logs/econagent_resume_156.log"
echo "  - 杀掉 session:     tmux kill-session -t $SESSION_NAME"
echo ""
echo "任务会在后台持续运行，即使 SSH 断开也不受影响"
echo ""

# 等待 3 秒后显示初始日志
sleep 3
echo "初始日志预览:"
echo "----------------------------------------"
tail -20 logs/econagent_resume_156.log 2>/dev/null || echo "日志文件尚未生成..."
echo "----------------------------------------"
echo ""
echo "使用 'tail -f logs/econagent_resume_156.log' 实时查看日志"
