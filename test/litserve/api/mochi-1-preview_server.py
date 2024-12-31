"""
https://huggingface.co/genmo/mochi-1-preview

from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/mochi-1-preview', local_dir='/mnt/data/llch/mochi-1-preview')
"""

"""
python3 ./demos/gradio_ui.py --model_dir /mnt/data/llch/mochi-1-preview/ --cpu_offload
export no_proxy="localhost,36.213.66.106,127.0.0.1,112.48.199.202,112.48.199.7"
demo.launch(server_name="0.0.0.0", server_port=16845)
"""
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import litserve as ls
import os
from dotenv import load_dotenv
import logging
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Mochi_1_P_API(ls.LitAPI):
    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        self.model_path = "/mnt/data/llch/mochi-1-preview"
        self.pipe = MochiPipeline.from_pretrained(
            self.model_path,
            # 高精度删除下面2行
            variant="bf16",
            torch_dtype=torch.bfloat16,
        )

        # Enable memory savings
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()

    def decode_request(self, request):
        prompt = request["prompt"]
        # prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
        return prompt

    def predict(self, inputs):
        prompt = inputs
        frames = self.pipe(prompt, num_frames=84).frames[0]
        out = export_to_video(frames, "mochi.mp4", fps=30)
        return out


if __name__ == "__main__":
    # export no_proxy="localhost,36.213.66.106,127.0.0.1,1.12.251.149"
    # python test/litserve/api/mochi-1-preview_server.py
    # nohup python test/litserve/api/mochi-1-preview_server.py > no_git_oic/mochi-1-preview_server.log &
    api = Mochi_1_P_API()
    server = ls.LitServer(api, accelerator="gpu", devices=1)
    server.run(port=int(os.getenv("MOCHI_PORT", 8123)))
