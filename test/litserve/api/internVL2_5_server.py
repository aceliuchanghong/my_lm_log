import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
import litserve as ls
import os
from dotenv import load_dotenv
import logging
import sys
from openai import OpenAI
from termcolor import colored
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import json

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
from test.litserve.api.quick_ocr_api_server import extract_entity, get_local_images


# https://huggingface.co/OpenGVLab/InternVL2_5-8B
def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL2_5API(ls.LitAPI):
    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        self.path = "/mnt/data/llch/InternVL2_5-8B/OpenGVLab/InternVL2_5-8B"
        self.device_map = split_model("InternVL2_5-8B")
        self.model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, trust_remote_code=True, use_fast=False
        )
        self.llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

    def decode_request(self, request):
        logger.info(colored(f"request recieved suc", "green"))
        # logger.info(colored(f"request:{request}", "green"))
        openai_msg = [el.dict() for el in request.messages]
        images_path = []
        local_images_path = []
        rule = {}
        prompt = ""
        user_prompt = ""
        # 遍历 openai_msg
        for msg in openai_msg:
            if isinstance(msg["content"], str):
                prompt += msg["content"]
                continue
            try:
                for content in msg["content"]:
                    if content["type"] == "image_url":
                        images_path.append(content["image_url"]["url"])
                    elif content["type"] == "text":
                        try:
                            # 尝试将文本内容转换为字典
                            text_dict = json.loads(content["text"])
                            rule.update(text_dict)
                        except json.JSONDecodeError:
                            # 如果转换失败，将文本内容存储为 prompt
                            prompt += content["text"]
            except Exception as e:
                logger.info(colored(f"no pics here", "green"))
        if len(images_path) > 0:
            local_images_path = get_local_images(images_path)
            logger.info(colored(f"pics recieved suc", "green"))
        try:
            user_prompt = (
                "提取"
                + rule["entity_name"]
                + (
                    ',其中output-example:"' + rule["entity_format"] + '"'
                    if len(rule["entity_format"]) > 1
                    else ""
                )
                + (
                    ',output可能的regex_pattern为:"'
                    + rule["entity_regex_pattern"]
                    + '"'
                    if len(rule["entity_regex_pattern"]) > 1
                    else ""
                )
            )
        except Exception as e:
            pass
        logger.info(
            colored(
                f"\nlocal_images_path:\n{local_images_path}\nuser_prompt:\n{user_prompt}\nrule:\n{rule}\nprompt:\n{prompt}",
                "green",
            )
        )
        return local_images_path, user_prompt, rule, prompt

    def predict(self, inputs):
        try:
            # 图片链接,提取关键信息的prompt,规则,还原markdown等prompt信息
            local_images_path, user_prompt, rule, prompt = inputs
            if len(prompt) > 0:
                user_prompt = prompt
                logger.info(colored(f"prompt:{user_prompt}", "green"))
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            pixel_values_list = [
                load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                for image_path in local_images_path  # 使用 local_images_path 提供的路径列表
            ]
            # 如果有接收到图片,那就去视觉提取
            if len(local_images_path) > 0:
                logger.info(colored(f"step1 vision start", "green"))
                question = f"<image>\n{user_prompt}"
                logger.info(colored(f"vision prompt:{question}", "green"))
                pixel_values = torch.cat(pixel_values_list, dim=0)
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True,
                )
                logger.info(colored(f"vision model result:{response}", "green"))
            else:
                logger.info(colored(f"step1 qwen start", "green"))
                response_llm = self.llm.chat.completions.create(
                    model=os.getenv("MODEL"),
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.5,
                )
                response = response_llm.choices[0].message.content
                # logger.info(colored(f"response_qwen:{response_llm}", "green"))
            # 如果只有规则的user_prompt, 还原markdown等prompt信息没用
            if len(prompt) == 0:
                logger.info(colored(f"step2 qwen extract start", "green"))
                ans = extract_entity(self.llm, rule, response)
                logger.info(colored(f"llm result:{ans}", "green"))

                response = {
                    "result": ans["result"],
                    "entity_name": ans["entity_name"],
                }
            logger.info(colored(f"final response:{response}", "green"))
            yield str(response)
        except Exception as e:
            logger.error(f"error:{e} \ninputs:{inputs}")
        finally:
            self.clean_memory(self.device)

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if (
            auth.scheme != "Bearer"
            or auth.credentials != "torch-yzgjhdxfxfyzdjhljsjed5h"
        ):
            raise HTTPException(status_code=401, detail="Authorization Failed")


if __name__ == "__main__":
    # export no_proxy="localhost,36.213.66.106,127.0.0.1,1.12.251.149"
    # python test/litserve/api/internVL2_5_server.py
    # nohup python test/litserve/api/internVL2_5_server.py > no_git_oic/internVL2_5_server.log &
    api = InternVL2_5API()
    # spec=ls.OpenAISpec()
    server = ls.LitServer(
        api, accelerator="gpu", devices=1, track_requests=True, spec=ls.OpenAISpec()
    )
    server.run(port=int(os.getenv("MOLMO_PORT")))
