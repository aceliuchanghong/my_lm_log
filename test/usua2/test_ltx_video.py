"""
# Model Download
from modelscope import snapshot_download

model_dir = snapshot_download(
    "AI-ModelScope/LTX-Video",
    local_dir="/mnt/data/llch/LTX-Video/checkpoints",
)
model_dir = snapshot_download(
    "CloseGPT/PixArt-XL-2-1024-MS",
    local_dir="/mnt/data/llch/CloseGPT/PixArt-XL-2-1024-MS",
)

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


conda create -n ltx_video python=3.10
pip install imageio


https://github.com/Lightricks/LTX-Video
python inference.py --ckpt_path '../ltx_checkpoint' --prompt "PROMPT" --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED
python inference.py --ckpt_path '../ltx_checkpoint' --prompt "PROMPT" --input_image_path IMAGE_PATH --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED
"""

"""
python inference.py --ckpt_path '../ltx_checkpoint' \
    --prompt "a read hair woman dancing in a yard,under the bright sun.flowers around" \
    --height 480 \
    --width 704 \
    --num_frames 121 \
    --seed 123456

python inference.py --ckpt_path '../ltx_checkpoint' \
    --prompt "a read hair woman dancing in a yard,under the bright sun.flowers around her" \
    --input_image_path IMAGE_PATH \
    --height 480 \
    --width 704 \
    --num_frames 121\
    --seed 123456
"""
