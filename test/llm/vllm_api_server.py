from modelscope import snapshot_download

# source activate vllm
# cd /mnt/data/llch/vllm
model_dir = snapshot_download(
    "Qwen/Qwen2.5-72B-Instruct", cache_dir="/mnt/data/qwen2.5-72-instruct"
)

# https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#extra-parameters-for-chat-api
# https://github.com/datawhalechina/self-llm/blob/master/models/Qwen2.5/03-Qwen2.5-7B-Instruct%20vLLM%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md
# https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html
# --host --port --api-key
# python -m vllm.entrypoints.openai.api_server --model /mnt/data/qwen2.5-72-instruct/Qwen2.5-72B-Instruct  --served-model-name Qwen2.5 --max-model-len=32768
# vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4

new_path = "qwen2.5-72-instruct/Qwen/Qwen2___5-72B-Instruct/"
port = 11433
host = "0.0.0.0"
api_key = "torch-elskenrgvoiserngviopsejrmoief"
"""
nohup python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/qwen2.5-72-instruct/Qwen/Qwen2___5-72B-Instruct/ \
    --served-model-name Qwen2.5 \
    --max-model-len 16384 \
    --tensor-parallel-size 4 \
    --port 11433 \
    --host 0.0.0.0 \
    --api-key torch-elskenrgvoiserngviopsejrmoief \
    --dtype auto \
    --kv-cache-dtype auto \
    --gpu_memory_utilization 0.98 \
    --task generate \
    > vllm_start_from_1128.log 2>&1 &
"""
