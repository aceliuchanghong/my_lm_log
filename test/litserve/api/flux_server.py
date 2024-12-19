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


class FluxLitAPI(ls.LitAPI):
    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        # Load the model
        torch_dtype = torch.bfloat16
        FLUX_1_schnell_model_path = "/mnt/data/llch/Flux_Models"
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
        prompt = request["prompt"]
        width = request["width"]
        height = request["height"]
        num_inference_steps = request["num_inference_steps"]
        guidance_scale = request["guidance_scale"]
        return prompt, width, height, num_inference_steps, guidance_scale

    def predict(self, params_receieved):
        prompt, width, height, num_inference_steps, guidance_scale = params_receieved
        logger.info(colored(f"params_receieved:{params_receieved}", "green"))
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(int(time.time())),
            guidance_scale=guidance_scale,
        ).images[0]

        return image

    def encode_response(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return Response(
            content=buffered.getvalue(), headers={"Content-Type": "image/png"}
        )

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
        api, accelerator="gpu", devices=1, track_requests=True, spec=ls.OpenAISpec()
    )
    server.run(port=int(os.getenv("FLUX_PORT")))
