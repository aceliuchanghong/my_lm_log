from io import BytesIO
from fastapi import Response
import torch
import time
import litserve as ls
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import time
import sys
import base64
import json

"""
source activate base
from modelscope import snapshot_download
model_dir = snapshot_download('black-forest-labs/FLUX.1-schnell',cache_dir="/mnt/data/llch/Flux_Models")
model_dir = snapshot_download('AI-ModelScope/clip-vit-large-patch14',cache_dir="/mnt/data/llch/clip-vit")
"""
load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)


class FluxLitAPI(ls.LitAPI):
    # @staticmethod
    # def clean_memory(device):
    #     import gc

    #     if torch.cuda.is_available():
    #         with torch.cuda.device(device):
    #             torch.cuda.empty_cache()
    #             torch.cuda.ipc_collect()
    #     gc.collect()

    def setup(self, device):
        # Load the model
        torch_dtype = torch.bfloat16
        FLUX_1_schnell_model_path = (
            "/mnt/data/llch/Flux_Models/black-forest-labs/FLUX___1-schnell"
        )
        clip_vit_large_patch14_model_path = (
            "/mnt/data/llch/clip-vit/AI-ModelScope/clip-vit-large-patch14"
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            FLUX_1_schnell_model_path, subfolder="scheduler"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            clip_vit_large_patch14_model_path, torch_dtype=torch_dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            clip_vit_large_patch14_model_path, torch_dtype=torch_dtype
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            FLUX_1_schnell_model_path,
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            FLUX_1_schnell_model_path, subfolder="tokenizer_2", torch_dtype=torch_dtype
        )
        vae = AutoencoderKL.from_pretrained(
            FLUX_1_schnell_model_path, subfolder="vae", torch_dtype=torch_dtype
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            FLUX_1_schnell_model_path, subfolder="transformer", torch_dtype=torch_dtype
        )

        """
        quantize to 8-bit to fit on an L4
        不想量化直接注释
        freeze: 用于冻结模型的参数，使其在训练或推理过程中不更新
        quantize: 用于将模型的权重量化到指定的类型
        qfloat8: 一种8位浮点数的量化类型
        """
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        self.pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self.pipe.text_encoder_2 = text_encoder_2
        self.pipe.transformer = transformer
        self.pipe.enable_model_cpu_offload()

    def decode_request(self, request):
        # Extract prompt from request
        model = request.get("prompt", "dall-e-3")
        prompt = request.get(
            "prompt",
            "an old black robot sitting in a chair painting a picture on an easel of a futuristic cityscape, pop art",
        )
        size = request.get("size", "512x512")
        width, height = map(int, size.split("x"))
        n = request.get("n", 1)
        num_inference_steps = request.get("num_inference_steps", 4)
        guidance_scale = request.get("guidance_scale", 3.5)
        return model, prompt, width, height, n, num_inference_steps, guidance_scale

    def predict(self, params_receieved):
        # 默认值
        model, prompt, width, height, n, num_inference_steps, guidance_scale = (
            params_receieved
        )
        logger.info(colored(f"params_receieved:{params_receieved}", "green"))

        start_time = time.time()
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(int(time.time())),
            guidance_scale=guidance_scale,
        ).images[0]
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"图片生成耗时: {elapsed_time:.2f}秒")

        return image

    def encode_response(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # 构建 data URI
        data_uri = f"data:image/png;base64,{img_base64}"
        response_data = {"created": int(time.time()), "data": [{"url": data_uri}]}
        return json.dumps(response_data)

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if (
            auth.scheme != "Bearer"
            or auth.credentials != "torch-yzgjhdxfxfyzdjhljsjed5h"
        ):
            raise HTTPException(status_code=401, detail="Authorization Failed")


if __name__ == "__main__":
    """
    export no_proxy="localhost,36.213.66.106,127.0.0.1,1.12.251.149"
    python test/litserve/api/flux_server.py
    nohup python test/litserve/api/flux_server.py > no_git_oic/flux_server.log &
    """
    api = FluxLitAPI()
    server = ls.LitServer(
        api, accelerator="gpu", devices=1, api_path="/v1/images/generations"
    )
    server.run(port=int(os.getenv("FLUX_PORT")))
