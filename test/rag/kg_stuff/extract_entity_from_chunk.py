import os
from dotenv import load_dotenv
import logging
import sys
import time
from termcolor import colored


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)
from test.rag.kg_stuff.contextual_rag import (
    get_basic_info_about_the_whole_doc,
    get_contextual_rag,
    get_contextual_rag_combine,
)
from z_utils.get_ai_tools import my_tools

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_entity(chunks, client):
    for chunk in chunks:
        response = client.chat.completions.create(
            model=os.getenv("SMALL_MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": os.getenv("KG_SYSTEM_PROMPT").format(
                        ASPECT=domain, WHOLE_DOCUMENT=basic_info, CHUNK_CONTENT=chunk
                    ),
                },
                {
                    "role": "user",
                    "content": f"用户输入:\n{chunk}",
                },
            ],
            temperature=0.2,
        )
        temp = response.choices[0].message.content
        print(f"{temp}")


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/extract_entity_from_chunk.py
    md_file = "no_git_oic/60a7c6be-b796-4ac6-bab8-e867abfa2865/auto/60a7c6be-b796-4ac6-bab8-e867abfa2865_tsr.md"
    ai_tools = my_tools()

    start_time = time.time()
    basic_info, domain, _, _ = get_basic_info_about_the_whole_doc(md_file, ai_tools.llm)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        colored(
            f"\n文档基本信息:{basic_info[:80]}\n文档主题:{domain}\n获取文档信息耗时: {elapsed_time:.2f}秒",
            "green",
        )
    )

    start_time = time.time()
    new_chunks = get_contextual_rag(md_file, ai_tools.llm)
    contextual_chunks = get_contextual_rag_combine(md_file, new_chunks)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        colored(
            f"\n文档chunks:\n{contextual_chunks}\n获取文档chunk耗时: {elapsed_time:.2f}秒",
            "light_blue",
        )
    )

    start_time = time.time()
    get_entity(contextual_chunks, ai_tools.llm)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        colored(
            f"\n获取文档KG耗时: {elapsed_time:.2f}秒",
            "light_yellow",
        )
    )
