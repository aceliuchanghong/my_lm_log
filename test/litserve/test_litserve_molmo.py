import os
import litserve as ls
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

model_path = '/mnt/data/llch/molmo/Molmo-7B-D-0924'


class MolmoLitAPI(ls.LitAPI):
    def setup(self, device):
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

    def decode_request(self, request):
        image_path = request["image_path"]
        text_input = request["text_input"]
        if os.path.isfile(image_path):
            # 如果路径是本地文件
            image = Image.open(image_path)
        else:
            # 如果路径是URL
            image = Image.open(requests.get(image_path, stream=True).raw)
        return image, text_input

    def predict(self, inputs):
        image, text = inputs
        processor_inputs = self.processor.process(
            images=[image],
            text=text
        )
        processor_inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in processor_inputs.items()}
        output = self.model.generate_from_batch(
            processor_inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )
        generated_tokens = output[0, processor_inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return {"output": generated_text}

    def encode_response(self, output):
        return {"output": output["output"]}


if __name__ == "__main__":
    api = MolmoLitAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=1)
    server.run(port=8927)
