import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import sys
import time
import json
import concurrent.futures
from termcolor import colored

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
from test.rag.kg_stuff.com_rule_01_fsr import read_docx
from z_utils.get_ai_tools import my_tools
from z_utils.get_text_chunk import chunk_by_LCEL
from test.rag.kg_stuff.prompt import *


def parse_to_list(kg_final_result: str):
    # 将单引号替换为双引号以符合 JSON 格式
    formatted_result = kg_final_result.replace("'", '"')

    # 使用 json.loads 将字符串转换为 Python 列表
    try:
        parsed_list = json.loads(formatted_result)
    except json.JSONDecodeError:
        logger.error("ERR:解析失败，请检查输入格式是否正确。")
        return [
            {"head": "解析错误头", "relation": "list生成失败", "tail": "解析错误尾"}
        ]

    return parsed_list


def process_content(
    content,
    structs_list,
    previous_content,
    ai_tools,
    model,
    aspect,
    page_number,
    temperature=0.2,
):
    logger.info(colored(f"handling the {page_number}th chunk", "green"))
    messages = [
        {"role": "system", "content": structure_json_prompt.format(ASPECT=aspect)},
        {
            "role": "user",
            "content": f"主要参考的当前文档的结构树!!:\n{structs_list[page_number]}",
        },
        {
            "role": "user",
            "content": f"可以用作参考的上页文档部分内容:\n{previous_content}",
        },
        {"role": "user", "content": f"当前页面的详细内容:\n{content}"},
    ]

    # 初次 LLM 处理
    response = ai_tools.llm.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    result = response.choices[0].message.content
    logger.debug(f"json:\n{result}")
    # 避免多线程太多请求时出现失败或超时
    time.sleep(5)

    # JSON 转换
    conversion_messages = [
        {"role": "system", "content": structure_kg_prompt},
        {
            "role": "user",
            "content": f"主要参考的当前文档的结构树!!:\n{structs_list[page_number]}",
        },
        {"role": "user", "content": f"待转化的json内容:\n{result}"},
    ]
    # logger.info(f"conversion_messages:{conversion_messages}")
    response = ai_tools.llm.chat.completions.create(
        model=model, messages=conversion_messages, temperature=temperature
    )
    final_result = response.choices[0].message.content
    final_result_list = parse_to_list(final_result)
    time.sleep(5)

    # 添加必要的内容
    for res in final_result_list:
        res["id"] = page_number
        res["content"] = content
    logger.debug(f"final_result_list:\n{final_result_list}")

    return final_result_list


def process_docx_files_kg(
    docx_file, ai_tools, model, chunk_size, chunk_overlap, fsr, aspect
):
    structs_list = fsr
    content_read = chunk_by_LCEL(
        read_docx(docx_file), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    previous_content = "第一页,暂无内容"
    kg_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for page_number, content in enumerate(content_read, start=1):
            futures.append(
                executor.submit(
                    process_content,
                    content,
                    structs_list,
                    previous_content,
                    ai_tools,
                    model,
                    aspect,
                    page_number,
                )
            )
            previous_content = content  # 更新上一部分内容
            time.sleep(2)
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="kg生成中...",
        ):
            try:
                kg_list.extend(future.result())
            except Exception as e:
                logger.error(f"Error processing content: {e}")
    return kg_list


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/com_rule_02_kg.py
    docx_path = "no_git_oic/com_rule_start/TE-MF-B011培训管理制度V6.1-20230314.docx"
    ai_tools = my_tools()
    chunk_size = int(os.getenv("chunk_size"))
    chunk_overlap = int(os.getenv("chunk_overlap"))
    model = os.getenv("MODEL")
    aspect = "公司培训管理制度"
    fsr = [
        "公司培训管理制度",
        "公司培训管理制度\n├── 公司电子管理文件\n├── 培训管理制度\n├── TE-MF-B011",
        "公司培训管理制度\n├── 公司电子管理文件\n├── 培训管理制度\n│   ├── 目的\n│   ├── 范围\n│   ├── 职责\n├── TE-MF-B011",
        "公司培训管理制度\n├── 公司电子管理文件\n├── 培训管理制度\n│   ├── 目的\n│   ├── 范围\n│   ├── 职责\n├── TE-MF-B011-4",
        "公司培训管理制度\n├── 公司电子管理文件\n├── 培训管理制度\n│   ├── 目的\n│   ├── 范围\n│   ├── 职责\n├── TE-MF-B011-5",
    ]
    start_time = time.time()
    kg_list = process_docx_files_kg(
        docx_path, ai_tools, model, chunk_size, chunk_overlap, fsr, aspect
    )
    logger.info(f"{kg_list}")
    elapsed_time = time.time() - start_time
    logger.info(f"kg生成耗时: {elapsed_time:.2f}秒")
    with open("no_git_oic/fsr_kg.txt", "w") as f:
        f.write("[" + "\n".join(str(item) for item in kg_list) + "]")
