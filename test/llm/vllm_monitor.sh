#!/bin/bash
############################################################
# vllm 服务健康检测
# lch-20250102
# chmod +x vllm_monitor.sh
# nohup /mnt/data/llch/vllm/vllm_monitor.sh > /mnt/data/llch/vllm/monitor.log 2>&1 &
# /mnt/data/llch/vllm/
# ps -ef | grep vllm_monitor
############################################################
# 定义变量
LOG_DIR="/mnt/data/llch/vllm"
PID_FILE="$LOG_DIR/vllm.pid"
API_KEY="torch-elskenrgvoiserngviopsejrmoief"
MODEL_DIR="/mnt/data/qwen2.5-72-instruct/Qwen/Qwen2___5-72B-Instruct/"
MODEL_NAME="Qwen2.5"
MAX_MODEL_LEN="16384"
TENSOR_PARALLEL_SIZE="4"
PORT="11433"
HOST="0.0.0.0"
GPU_MEMORY_UTILIZATION="0.98"
CONDA_ENV_NAME="vllm"

# 获取当前日期
DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/${DATE}_vllm.log"

# 检查 vllm 是否在运行
is_vllm_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" > /dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

# 启动 vllm 服务
start_vllm() {
    # 激活 Conda 虚拟环境
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    if [ $? -eq 0 ]; then
        echo "$(date): Conda environment $CONDA_ENV_NAME activated successfully." >> "$LOG_FILE"
    else
        echo "$(date): Failed to activate Conda environment $CONDA_ENV_NAME." >> "$LOG_FILE"
        return 1
    fi

    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --port "$PORT" \
        --host "$HOST" \
        --api-key "$API_KEY" \
        --dtype auto \
        --kv-cache-dtype auto \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --task generate \
        > "$LOG_FILE" 2>&1 &
    if [ $? -eq 0 ]; then
        echo $! > "$PID_FILE"
        echo "$(date): vllm service started successfully. PID: $!" >> "$LOG_FILE"
    else
        echo "$(date): Failed to start vllm service." >> "$LOG_FILE"
    fi
}

# 主循环
while true; do
    if is_vllm_running; then
        echo "$(date): vllm is running. PID: $(cat $PID_FILE)" >> "$LOG_FILE"
    else
        echo "$(date): vllm is not running. Restarting..." >> "$LOG_FILE"
        start_vllm
    fi
    sleep 300
done
