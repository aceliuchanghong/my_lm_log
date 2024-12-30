"""
https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuan_video
https://huggingface.co/tencent/HunyuanVideo
https://github.com/Tencent/HunyuanVideo/blob/main/README_zh.md

from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/HunyuanVideo', local_dir='/mnt/data/llch/HunyuanVideo')

model_dir = snapshot_download('AI-ModelScope/llava-llama-3-8b-v1_1-transformers',local_dir='/mnt/data/llch/HunyuanVideo/text_encoder')
model_dir = snapshot_download('AI-ModelScope/clip-vit-large-patch14',local_dir='/mnt/data/llch/HunyuanVideo/text_encoder_2')
"""

"""
cd HunyuanVideo

uv run sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "a cat is running, realistic." \
    --flow-reverse \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results \
    --model-base /mnt/data/llch/HunyuanVideo \
    --dit-weight /mnt/data/llch/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
"""
