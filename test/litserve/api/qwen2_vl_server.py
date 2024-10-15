import litserve as ls
import os
from dotenv import load_dotenv
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


class QwenVLAPI(ls.LitAPI):
    def setup(self, device):
        # 加载模型和处理器
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.model_path = os.getenv("QWEN2_VL_MODEL_PATH")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.device = device

    def decode_request(self, request):
        # 处理传入的消息，包括多张图片
        messages = request["messages"]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def predict(self, inputs):
        # 模型推理
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def encode_response(self, output):
        # 返回模型的生成结果
        return {"output": output}


# STEP 2: START THE SERVER
if __name__ == "__main__":
    api = QwenVLAPI()
    server = ls.LitServer(api, accelerator="auto", devices=1)
    server.run(port=int(os.getenv("QWEN2_VL_PORT")))
