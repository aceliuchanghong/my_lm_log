import os
from dotenv import load_dotenv
import logging
import sys
import time


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)
from test.llm.test_llm_link import get_entity_result
from test.rag.test_chunk import split_text
from z_utils.get_ai_tools import my_tools
from z_utils.get_text_chunk import chunk_by_LCEL

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/simple_kg_01.py
    start_time = time.time()
    ai_tools = my_tools()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"AI初始化耗时: {elapsed_time:.2f}秒")

    # md_file = "no_git_oic/283f6c46-b947-4735-98c9-6e12c9c82aee/auto/283f6c46-b947-4735-98c9-6e12c9c82aee_tsr.md"
    md_file = "no_git_oic/1db87098-3f2a-4516-8307-e0517a7ec98e/auto/1db87098-3f2a-4516-8307-e0517a7ec98e_tsr.md"
    start_time = time.time()
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = split_text(content)
    for i, chunk in enumerate(chunks, start=1):
        print(f"split_text-the {i}:\n{chunk}")

    output = chunk_by_LCEL(md_file, chunk_size=700, chunk_overlap=300)
    for i, chunk in enumerate(output, start=1):
        print(f"chunk_by_LCEL the {i}:\n{chunk}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"文件读取耗时: {elapsed_time:.2f}秒")

    # start_time = time.time()
    # domain = get_entity_result(
    #     ai_tools.llm, content, system_prompt=os.getenv("GENERATE_DOMAIN_PROMPT")
    # )
    # try:
    #     result = domain.get("domain", "企业产品信息相关")
    # except Exception as e:
    #     result = "企业产品信息相关"
    # print(f"{result}")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # logger.info(f"获取领域耗时: {elapsed_time:.2f}秒")
