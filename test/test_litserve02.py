import litserve as ls
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

model_path = '/mnt/data/llch/molmo/Molmo-7B-D-0924'


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model1 = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.model2 = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

    def decode_request(self, request):
        # Extract file from request
        return (request["content"], request["prompt"])

    def predict(self, x):
        # Generate the upscaled image
        inputs = self.model1.process(
            images=[Image.open(input[0]).convert('RGB')],
            text="Describe this image."
        )
        inputs = {k: v.to(self.model2.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model2.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.model1.tokenizer
        )
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.model1.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=1)
    server.run(port=8927)
