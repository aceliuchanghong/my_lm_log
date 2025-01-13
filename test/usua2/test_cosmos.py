"""
# Model Download
from modelscope import snapshot_download

model_dir = snapshot_download(
    "AI-ModelScope/Cosmos-1.0-Tokenizer-CV8x8x8",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Cosmos-1.0-Tokenizer-CV8x8x8",
)

model_dir = snapshot_download(
    "AI-ModelScope/Cosmos-1.0-Prompt-Upsampler-12B-Text2World",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Cosmos-1.0-Prompt-Upsampler-12B-Text2World",
)

model_dir = snapshot_download(
    "AI-ModelScope/Cosmos-1.0-Diffusion-14B-Text2World",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Cosmos-1.0-Diffusion-14B-Text2World",
)

model_dir = snapshot_download(
    "AI-ModelScope/Cosmos-1.0-Diffusion-14B-Video2World",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Cosmos-1.0-Diffusion-14B-Video2World",
)

model_dir = snapshot_download(
    "AI-ModelScope/Cosmos-1.0-Autoregressive-13B-Video2World",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Cosmos-1.0-Autoregressive-13B-Video2World",
)

model_dir = snapshot_download(
    "AI-ModelScope/Cosmos-1.0-Autoregressive-12B",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Cosmos-1.0-Autoregressive-12B",
)

model_dir = snapshot_download(
    "AI-ModelScope/Cosmos-1.0-Guardrail",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Cosmos-1.0-Guardrail",
)

model_dir = snapshot_download(
    "LLM-Research/Pixtral-12B-2409",
    local_dir="/mnt/data/llch/Cosmos/checkpoints/Pixtral-12B-2409",
)

---

# Structure
checkpoints/
├── Cosmos-1.0-Diffusion-7B-Text2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-14B-Text2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-7B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-14B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Tokenizer-CV8x8x8
│   ├── decoder.jit
│   ├── encoder.jit
│   └── mean_std.pt
├── Cosmos-1.0-Prompt-Upsampler-12B-Text2World
│   ├── model.pt
│   └── config.json
├── Pixtral-12B
│   ├── model.pt
│   ├── config.json
└── Cosmos-1.0-Guardrail
    ├── aegis/
    ├── blocklist/
    ├── face_blur_filter/
    └── video_content_safety_filter/

---

# Pixtral-12B-2409 fix
vi cosmos1/scripts/convert_pixtral_ckpt.py
# snapshot_download(
#    repo_id=repo_id,
#    allow_patterns=["params.json", "consolidated.safetensors"],
#    local_dir=pixtral_ckpt_dir,
#    local_dir_use_symlinks=False,
#)

from cosmos1.scripts.convert_pixtral_ckpt import convert_pixtral_checkpoint
convert_pixtral_checkpoint(
    checkpoint_dir="checkpoints",
    checkpoint_name="Pixtral-12B",
    vit_type="pixtral-12b-vit",
)

source activate cosmos
(其路径:/usr/local/cuda/include/cudnn.h,先执行:export CPATH=/usr/local/cuda/include:$CPATH )
pip install transformer_engine[pytorch] attrs opencv-python einops pynvml

import argparse
import os
import torch

from cosmos1.models.diffusion.inference.inference_utils import add_common_arguments, check_input_frames, validate_args
from cosmos1.models.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldGenerationPipeline, DiffusionText2WorldGenerationPipeline
from cosmos1.utils import log, misc
from cosmos1.utils.io import read_prompts_from_file, save_video


PROMPT="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-14B-Text2World \
    --prompt "$PROMPT" \
    --offload_prompt_upsampler \
    --video_save_name Cosmos-1.0-Diffusion-14B-Text2World
"""
