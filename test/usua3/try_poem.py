import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import json
import zhconv  # pip install zhconv
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

from z_utils.get_json import parse_and_check_json_markdown


def transform2sim(list_or_string, tran_type="simple"):
    if isinstance(list_or_string, list):
        # 如果是列表，遍历列表中的每个字符串并进行转换
        new_zh = [
            zhconv.convert(item, "zh-hans" if tran_type == "simple" else "zh-hant")
            for item in list_or_string
        ]
    else:
        # 如果是单个字符串，直接进行转换
        new_zh = zhconv.convert(
            list_or_string, "zh-hans" if tran_type == "simple" else "zh-hant"
        )
    return new_zh


def get_poem(file_path):
    """
    从指定的 JSON 文件中读取诗歌数据，并返回每首诗歌的作者、标题和内容。

    :param file_path: JSON 文件路径
    :return: 包含 author, title, paragraphs 的字典列表
    """
    poems = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            datas = json.load(file)

        for i, data in enumerate(datas):
            author = data.get("author", "误道者")
            title = data.get("title", "大道争锋")
            paragraphs = data.get("paragraphs", ["愿起一剑杀万劫"])
            id = data.get("id", "wrong-id")

            poems.append(
                {
                    "id": id,
                    "author": transform2sim(author),
                    "title": transform2sim(title),
                    "paragraphs": transform2sim(paragraphs),
                }
            )
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")

    return poems


def get_embedded_poem(embeddings, poems):
    """
    从诗歌列表中提取诗歌内容，然后使用 embeddings 对诗歌内容进行嵌入。

    :param embeddings: 嵌入模型
    :param poems: 诗歌列表
    :return: 嵌入的诗歌列表
    """

    paragraphs = poems["colloquial_prose"]
    embedded_paragraphs = embeddings.embed_documents(paragraphs)

    poems["embedded_colloquial_prose"] = embedded_paragraphs

    return poems


class ColloquialProse(BaseModel):
    colloquial_prose: str


def trans_poem(client, poems: Dict) -> Dict:
    """
    将古诗转换为白话文

    :param client: 客户端对象，用于调用API
    :param poems: 包含古诗段落的字典
    :return: 包含白话文翻译的字典
    """
    fill_prompt = """Goal:帮我将古诗转为白话文
    Instruction:
        1.输出流畅优美的白话散文
        2.json格式输出
    Input:{poem}
    Output-Example:{{'colloquial_prose':'...'}}
    """

    def get_translation(prompt: str) -> str:
        """
        调用API获取翻译结果

        :param prompt: 输入的提示文本
        :return: 翻译后的白话文
        """
        try:
            response = client.beta.chat.completions.parse(
                model=os.getenv("SMALL_MODEL", "qwen2.5:7b-instruct-fp16"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format=ColloquialProse,
            )
            trans = parse_and_check_json_markdown(
                response.choices[0].message.content, ["colloquial_prose"]
            )
            return trans.get("colloquial_prose", "翻译错误")
        except Exception as e:
            logger.error(f"翻译过程中发生错误: {e}")
            return "翻译错误"

    colloquial_prose = []
    poems_paragraphs = poems.get("paragraphs", [])

    for paragraph in poems_paragraphs:
        prompt = fill_prompt.format(poem=paragraph)
        logger.debug(colored(f"prompt: {prompt}", "green"))
        translation = get_translation(prompt)
        colloquial_prose.append(translation)

    poems["colloquial_prose"] = colloquial_prose
    return poems


def construct_new_path(original_json_path, new_base_path):
    """Construct the new path based on the original path and the new base path."""
    this_json_path = Path(original_json_path)
    parts = this_json_path.parts
    start_index = parts.index("chinese-poetry") + 1
    inter_path = Path(*parts[start_index:-1])
    file_name = parts[-1]
    return Path(new_base_path) / inter_path / file_name


def save_new_json(
    original_json_path, client, embeddings, new_path="../chinese-poetry/trans_and_emb"
):
    new_json_path = construct_new_path(original_json_path, new_path)
    new_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 临时文件路径
    temp_json_path = new_json_path.with_suffix(".tmp")

    try:
        with open(new_json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    # 提取 existing_data 中已有的 poem id
    existing_ids = {poem["id"] for poem in existing_data}

    poems = get_poem(original_json_path)

    def process_poem(poem):
        # 如果 poem["id"] 已经存在，则跳过处理
        if poem["id"] in existing_ids:
            return None
        new_poem = trans_poem(client, poem)
        poem_with_vec = get_embedded_poem(embeddings, new_poem)
        return poem_with_vec

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_poem, poem) for poem in poems]
        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(poems), desc="Processing poems")
        ):
            result = future.result()
            if result is not None:  # 只添加非 None 的结果
                existing_data.append(result)

            # 每处理 100 个 poem 保存一次进度
            if i % 100 == 0:
                with open(temp_json_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)

    # 最终保存
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    # 重命名临时文件为最终文件
    os.replace(temp_json_path, new_json_path)

    return len(existing_data)


if __name__ == "__main__":
    # python test/usua3/try_poem.py
    # export no_proxy="localhost,127.0.0.1"
    json_file_path = "../chinese-poetry/全唐诗/error/poet.song.1000.json"
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMB_MODEL", "bge-m3"), base_url=os.getenv("EMB_BASE_URL")
    )
    client = OpenAI(
        api_key=os.getenv("API_KEY"), base_url=os.getenv("OLLAMA_CHAT_BASE_URL")
    )

    # poems = get_poem(json_file_path)
    # logger.info(colored(f"诗歌信息开始:{poems[5]}", "green"))
    # new_poems = trans_poem(client, poems[5])
    # logger.info(colored(f"诗歌翻译:{new_poems}", "green"))
    # poem_with_vec = get_embedded_poem(embeddings, new_poems)
    # logger.info(colored(f"向量转化:{len(poem_with_vec["colloquial_prose"])}", "green"))

    nums = save_new_json(json_file_path, client, embeddings)
    logger.info(colored(f"新的诗歌数量:{nums}", "green"))
