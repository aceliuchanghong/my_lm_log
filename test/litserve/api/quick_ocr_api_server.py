import os
import litserve as ls
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from dotenv import load_dotenv
import requests
from openai import OpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import logging
import time
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import base64
from PIL import Image
import io
from hashlib import md5
import random
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from test.ocr.test_surya import polygon_to_markdown
from test.ocr.test_combine_ocr import create_textline_from_data
from test.llm.test_llm_link import get_entity_result
from z_utils.rotate2fix_pic import detect_text_orientation
from z_utils.get_text_chunk import chunk_by_LCEL

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_entity(llm, rule, all_text, retriever=None):
    """
    提取指定的实体信息。

    :param llm: 语言模型实例，用于生成答案
    :param rule: 包含 entity_name, entity_format, entity_regex_pattern 的规则
    :param all_text: 要从中提取实体的文本
    :return: 提取的实体结果
    """
    entity = {}

    # 构建用户提示词
    user_prompt = (
        "提取"
        + rule["entity_name"]
        + (
            ',answer-value-example:"' + rule["entity_format"] + '"'
            if len(rule["entity_format"]) > 1
            else ""
        )
        + (
            ',answer-value-regex-pattern为:"' + rule["entity_regex_pattern"] + '"'
            if len(rule["entity_regex_pattern"]) > 1
            else ""
        )
    )

    # 从文本中提取信息
    basic_info = all_text
    if len(all_text) > int(os.getenv("long_ocr_result")):
        basic_info = "".join(
            [i.page_content for i in retriever.invoke(user_prompt)[:2]]
        )
        logger.info(f"emb结果:{basic_info}")
    ans = get_entity_result(
        llm, user_prompt, basic_info[: min(int(os.getenv("long_ocr_result")), 1800)]
    )
    logger.info(f"LLM回答结果:{ans}")

    if "answer" in ans and ans["answer"] != "DK":
        entity["result"] = ans["answer"]
    else:
        # 如果模型无法提取，尝试用正则匹配
        entity["result"] = match_entity_with_regex(rule, all_text)

    entity["entity_name"] = rule["entity_name"]
    return entity


def match_entity_with_regex(rule, all_text):
    """
    使用正则表达式从文本中匹配实体。

    :param rule: 包含 entity_regex_pattern 的规则
    :param all_text: 要从中匹配实体的文本
    :return: 匹配到的实体或 "DK"
    """
    if len(rule["entity_regex_pattern"]) > 1:
        match = re.search(rule["entity_regex_pattern"], all_text)
        if match:
            logger.info(f"regex: {match.group(0)}")
            return match.group(0)
    return "DK"


def download_image(image_url, save_dir):
    """Download an image from a URL and save it locally."""
    os.makedirs(save_dir, exist_ok=True)
    local_image_path = os.path.join(save_dir, os.path.basename(image_url))

    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(local_image_path, "wb") as out_file:
            out_file.write(response.content)
        return local_image_path
    else:
        raise ValueError(f"Failed to download image from {image_url}")


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def base64_to_image(base64_string, save_dir):
    # 去掉前缀，例如 "image/jpeg;base64,"
    base64_data = base64_string.split(",")[-1]
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))

    random_number = random.randint(1, 10000)
    name = compute_mdhash_id(base64_data[:100], prefix="pic_")
    image_name = name + str(random_number) + ".jpg"

    image_path = os.path.join(save_dir, image_name)
    logger.info(colored(f"base64 image_path:{image_path}\n", "green"))
    image.save(image_path)
    return image_path


def get_local_images(images_path):
    save_dir = os.path.join(os.getenv("upload_file_save_path"), "images")
    local_images_path = set()  # 使用集合来避免重复
    rotate_path = os.path.join(os.getenv("upload_file_save_path"), "rotate_pics")

    # 多线程下载和处理图片
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_image = {}
        for image in images_path:
            if image.startswith("http"):
                future_to_image[executor.submit(download_image, image, save_dir)] = (
                    image
                )
            elif image.startswith("data:image/"):
                future_to_image[executor.submit(base64_to_image, image, save_dir)] = (
                    image
                )
            elif os.path.isfile(image):
                local_images_path.add(image)
            else:
                local_images_path.add("get_local_images:wrong format")

        # 处理所有任务，无论是下载还是文字方向检测
        for future in as_completed(future_to_image):
            image = future_to_image[future]
            try:
                # 下载后进行文字方向检测
                downloaded_image = future.result()
                result_image = detect_text_orientation(downloaded_image, rotate_path)
                local_images_path.add(result_image)  # 使用集合的add方法
            except Exception as e:
                print(f"Error processing image {image}: {e}")
        logger.debug(f"local_images_path1:{list(local_images_path)}")

        # 对本地文件进行文字方向检测
        for image in images_path:
            if os.path.isfile(image):
                try:
                    result_image = detect_text_orientation(image, rotate_path)
                    local_images_path.add(result_image)  # 使用集合的add方法
                except Exception as e:
                    print(f"Error processing image {image}: {e}")
        logger.debug(f"local_images_path2:{list(local_images_path)}")

    return list(local_images_path)  # 返回列表格式


class QuickOcrAPI(ls.LitAPI):
    def ocr_image(self, tsr, local_image):
        single_result = {}
        single_result["file_name"] = os.path.basename(local_image)

        rapid_ocr_result, _ = self.ocr_engine(local_image)
        # ss = ""
        # for buck in ocr_result:
        #     ss += buck[1] + " "
        # if tsr == "tsr":
        #     table_html_str, table_cell_bboxes, elapse = self.table_engine(
        #         local_image, ocr_result
        #     )
        #     ss = table_html_str.replace("<html><body>", "").replace(
        #         "</body></html>", ""
        #     )
        text_lines = []
        if rapid_ocr_result is None:
            rapid_ocr_result = []
        for line in rapid_ocr_result:
            text_line = create_textline_from_data(line)
            text_lines.append(text_line)
        markdown0 = polygon_to_markdown(text_lines)
        # markdown1 = markdown0.splitlines()[: int(os.getenv("import_head_lines"))]
        markdown1 = markdown0.splitlines()
        rapid_ocr_markdown = "\n".join([text for text in markdown1 if len(text) > 0])
        logger.info(f"rapid_ocr_markdown:\n{rapid_ocr_markdown}")

        # ip = "127.0.0.1"
        # response = requests.post(
        #     f"http://{ip}:{os.getenv('SURYA_PORT')}/predict",
        #     json={
        #         "images_path": [
        #             os.path.join(os.getcwd(), local_image),
        #         ],
        #     },
        # )
        # import json

        # surya_ocr_result = json.loads(response.text)["output"][
        #     os.path.join(os.getcwd(), local_image)
        # ]
        # logger.info(f"surya_ocr_result:{surya_ocr_result}")

        # single_result["ocr_result"] = rapid_ocr_markdown + surya_ocr_result
        single_result["ocr_result"] = rapid_ocr_markdown

        try:
            save_dir = os.path.join(os.getenv("upload_file_save_path"), "images")
            rotate_path = os.path.join(
                os.getenv("upload_file_save_path"), "rotate_pics"
            )
            upload_image = os.path.join(save_dir, single_result["file_name"])
            rotate_image = os.path.join(rotate_path, single_result["file_name"])
            logger.info(f"starting to rm files...")
            os.remove(upload_image)
            os.remove(rotate_image)
        except Exception as e:
            logger.error(f"{e}\ndel:{upload_image} or {rotate_image} failed")

        return single_result

    def setup(self, device):
        self.table_engine = RapidTable(
            model_path=os.getenv("rapidocr_table_engine_model_path")
        )
        self.ocr_engine = RapidOCR()
        self.llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("EMB_MODEL"), base_url=os.getenv("EMB_BASE_URL")
        )

    def decode_request(self, request):
        logger.info(f"received request:\n{request}")
        images_path = request["images_path"]
        tsr = request.get("table", "normal")  # 只能tsr,normal
        rules = request["rule"]
        if tsr != "normal":
            tsr = "tsr"
        local_images_path = get_local_images(images_path)

        return tsr, local_images_path, rules

    def predict(self, inputs):
        tsr, local_images_path, rules = inputs
        start_time = time.time()
        ocr_output_result = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda img: self.ocr_image(tsr, img), local_images_path
            )
        for result in results:
            ocr_output_result.append(result)

        all_text = "".join([item["ocr_result"] for item in ocr_output_result])

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"OCR耗时: {elapsed_time:.2f}秒")

        start_time = time.time()
        entity_list = []

        retriever = None
        if len(all_text) > int(os.getenv("long_ocr_result")):
            text_doc = chunk_by_LCEL(all_text)
            vectorstore = InMemoryVectorStore.from_texts(
                text_doc,
                embedding=self.embeddings,
            )
            retriever = vectorstore.as_retriever()

        # 定义多线程处理单个规则的函数
        def process_single_rule(rule):
            logger.debug(colored(f"all_text:{all_text}", "green"))
            return extract_entity(self.llm, rule, all_text, retriever)

        # 使用 ThreadPoolExecutor 并行执行规则处理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_single_rule, rules))

        # 将结果加入实体列表
        for entity in results:
            entity_list.append(entity)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"LLM提取实体耗时: {elapsed_time:.2f}秒")

        return {"ocr_result": ocr_output_result, "entity_result": entity_list}

    def encode_response(self, output):
        return output["ocr_result"], output["entity_result"]


if __name__ == "__main__":
    # python test/litserve/api/quick_ocr_api_server.py
    # export no_proxy="localhost,10.6.6.113,127.0.0.1,1.12.251.149"
    # nohup python test/litserve/api/quick_ocr_api_server.py > no_git_oic/quick_ocr_api_server.log &
    api = QuickOcrAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=[1])
    server.run(port=int(os.getenv("quick_ocr_port")))
