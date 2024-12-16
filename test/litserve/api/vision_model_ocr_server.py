import os
import litserve as ls
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from openai import OpenAI
import logging
import sys
from PIL import Image

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)
from test.litserve.api.quick_ocr_api_server import extract_entity, get_local_images

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


class MolmoOcrLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model_path = os.getenv("MOLMO_MODEL_PATH")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

    def decode_request(self, request):
        images_path = request["images_path"]
        rule = request["rule"]
        local_images_path = get_local_images(images_path)
        user_prompt = (
            "提取"
            + rule["entity_name"]
            + (
                ",它的可能结果案例:" + rule["entity_format"]
                if len(rule["entity_format"]) > 1
                else ""
            )
            + (
                ",它的可能结果正则:" + rule["entity_regex_pattern"]
                if len(rule["entity_regex_pattern"]) > 1
                else ""
            )
        )
        logger.debug(f"local_images_path:\n{local_images_path}")
        logger.debug(f"user_prompt:\n{user_prompt}")
        return local_images_path, user_prompt, rule

    def predict(self, inputs):
        local_images_path, text, rule = inputs
        image_object = []
        for local_image in local_images_path:
            temp = Image.open(local_image)
            image_object.append(temp)
        processor_inputs = self.processor.process(images=image_object, text=text)
        processor_inputs = {
            k: v.to(self.model.device).unsqueeze(0) for k, v in processor_inputs.items()
        }
        output = self.model.generate_from_batch(
            processor_inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )
        generated_tokens = output[0, processor_inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        logger.info(f"vision model result:{generated_text}")

        ans = extract_entity(self.llm, rule, generated_text)
        logger.debug(f"llm_ans:{ans}")

        return {"result": ans["result"], "entity_name": ans["entity_name"]}

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    # python test/litserve/api/vision_model_ocr_server.py
    # export no_proxy="localhost,36.213.66.106,127.0.0.1"
    api = MolmoOcrLitAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=1)
    server.run(port=int(os.getenv("MOLMO_PORT")))
