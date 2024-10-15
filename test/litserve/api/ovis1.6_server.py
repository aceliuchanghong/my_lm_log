import litserve as ls
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
import os
from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


class Ovis16API(ls.LitAPI):
    def setup(self, device):
        model_path = os.getenv("OVIS_MODEL_PATH")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=8192,
            trust_remote_code=True,
            device_map=device,
            # device_map="auto",  # 由lit分配到多个卡上面,但是RuntimeError: Expected all tensors to be on the same device
        )
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def decode_request(self, request):
        images = []
        query = ""
        images_path = request["images_path"]
        for image_path in images_path:
            image = Image.open(image_path)
            images.append(image)
            query += "<image>\n"

        text = request["text"]
        query += text
        return query, images

    def predict(self, inputs):
        # 处理输入，进行推理，返回结果
        query, images = inputs
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [
            pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
            )
        ]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return output

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = Ovis16API()
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=[
            3,
        ],
    )
    server.run(port=int(os.getenv("OVIS_PORT")))
