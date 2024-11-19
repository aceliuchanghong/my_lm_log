from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import litserve as ls
import os
from dotenv import load_dotenv
import logging
import sys


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

from test.litserve.api.quick_ocr_api_server import download_image


class FlorenceAPI(ls.LitAPI):
    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        self.torch_dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            os.getenv("FLOWRENCE_MODEL_PATH"),
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(
            os.getenv("FLOWRENCE_MODEL_PATH"), trust_remote_code=True
        )
        self.prompt = "<DETAILED_CAPTION>"
        self.device = device

    def decode_request(self, request):
        images = {}
        files_name = request["images"]
        for file_path in files_name:
            if not os.path.isfile(file_path):
                file_path = download_image(
                    file_path,
                    os.path.join(os.getenv("upload_file_save_path"), "flowrence"),
                )
            image = Image.open(file_path).convert("RGB")
            images[os.path.basename(file_path)] = image
        logger.info(f"{images}")
        return images

    def predict(self, inputs):
        result = {}
        try:
            for name, image in inputs.items():
                inputs = self.processor(
                    text=self.prompt, images=image, return_tensors="pt"
                ).to(self.device, self.torch_dtype)
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text,
                    task=self.prompt,
                    image_size=(image.width, image.height),
                )
                caption_text = parsed_answer["<DETAILED_CAPTION>"].replace(
                    "The image shows ", ""
                )
                result[name] = caption_text
            return result
        except Exception as e:
            logger.error(f"error:{e}")
        finally:
            self.clean_memory(self.device)

    def encode_response(self, response):
        return response


if __name__ == "__main__":
    # python test/litserve/api/florence-2.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # nohup python test/litserve/api/florence-2.py > no_git_oic/florence.log &
    server = ls.LitServer(FlorenceAPI(), accelerator="auto", devices=[3])
    server.run(port=int(os.getenv("FLORENCE_PORT")))
