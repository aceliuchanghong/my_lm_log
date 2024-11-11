import os
from dotenv import load_dotenv
import logging
import sys
import time
import re
import concurrent.futures


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from test.llm.test_llm_link import get_entity_result
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


def get_basic_info_about_the_whole_doc(md_doc, client=None):

    with open(md_doc, "r") as file:
        content = file.read()

    # 正则表达式匹配一级、二级标题
    h1_titles = re.findall(r"^# (.+)", content, re.MULTILINE)
    h2_titles = re.findall(r"^## (.+)", content, re.MULTILINE)

    # 去除重复的标题
    h1_titles_unique = list(set(h1_titles))
    h2_titles_unique = list(set(h2_titles))

    basic_info = "一级标题:\n" if len(h1_titles_unique) > 0 else ""
    for h1 in h1_titles_unique:
        basic_info += h1 + "\n"
    basic_info += "\n---\n\n二级标题:\n" if len(h2_titles_unique) > 0 else ""
    for h2 in h2_titles_unique:
        basic_info += h2 + "\n"

    if len(basic_info) < 80:
        remaining_content = content[: 500 - len(basic_info)]
        basic_info += "\n---\n\n文档内容预览:\n" + remaining_content

    domain = ""
    entity_type_list = []
    relations_list = []
    if client is not None:
        result = get_entity_result(
            client, basic_info, system_prompt=os.getenv("GENERATE_DOMAIN_PROMPT")
        )
        try:
            domain = result.get("domain", "企业产品信息相关")
        except Exception as e:
            domain = "企业产品信息相关"

        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": os.getenv("KG_SYSTEM_PROMPT"),
                },
                {
                    "role": "user",
                    "content": f"文档领域:{domain}\n文档基本信息:\n{basic_info}",
                },
            ],
            temperature=0.2,
        )
        temp = response.choices[0].message.content

    return basic_info, domain, entity_type_list, relations_list


def get_contextual_rag(md_file, client):
    new_chunks = []
    basic_info, domain, _, _ = get_basic_info_about_the_whole_doc(md_file, client)

    content_read = chunk_by_LCEL(
        md_file,
        chunk_size=int(os.getenv("chunk_size")),
        chunk_overlap=int(os.getenv("chunk_overlap")),
    )

    # 定义生成每个并发请求的函数
    def process_chunk(chunk):
        response = client.chat.completions.create(
            model=os.getenv("SMALL_MODEL"),
            messages=[
                {
                    "role": "user",
                    "content": os.getenv("CONTEXTUAL_RAG_PROMPT").format(
                        ASPECT=domain, WHOLE_DOCUMENT=basic_info, CHUNK_CONTENT=chunk
                    ),
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk): chunk for chunk in content_read
        }
        for future in concurrent.futures.as_completed(future_to_chunk):
            new_chunks.append(future.result())

    return new_chunks


def get_contextual_rag_combine(md_file, new_chunks):
    content_chunk = chunk_by_LCEL(
        md_file,
        chunk_size=int(os.getenv("chunk_size")),
        chunk_overlap=int(os.getenv("chunk_overlap")),
    )
    contextual_chunks = []

    # 遍历content_chunk，为每个块添加前后文
    for i, chunk in enumerate(content_chunk):
        # 获取前一个和后一个总结，如果不存在则设为空字符串
        prev_summary = new_chunks[i - 1] if i - 1 >= 0 else ""
        next_summary = new_chunks[i + 1] if i + 1 < len(new_chunks) else ""

        # 组合上下文，创建新的chunk
        contextual_chunk = f"{prev_summary} {chunk} {next_summary}"
        contextual_chunks.append(contextual_chunk)

    return contextual_chunks


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/contextual_rag.py
    md_file = "no_git_oic/60a7c6be-b796-4ac6-bab8-e867abfa2865/auto/60a7c6be-b796-4ac6-bab8-e867abfa2865_tsr.md"
    ai_tools = my_tools()

    start_time = time.time()
    new_chunks = get_contextual_rag(md_file, ai_tools.llm)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"AI总结耗时: {elapsed_time:.2f}秒")

    contextual_chunks = get_contextual_rag_combine(md_file, new_chunks)
    print(f"{contextual_chunks}")
