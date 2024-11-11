import networkx as nx
import os
from dotenv import load_dotenv
import logging
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.get_text_chunk import chunk_by_LCEL

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

"""
langchain常用:https://python.langchain.com/api_reference/langchain/index.html
GraphRAG中模板提示词中文版:https://blog.csdn.net/weixin_46248339/article/details/140530006 https://blog.csdn.net/weixin_53141214/article/details/141859710
用GPT-3.5构建知识图谱:https://mp.weixin.qq.com/s/uq7ZWHsVGIdEvrAodvmk4A
Late Chunking:https://mp.weixin.qq.com/s/I69YEZZl9EGtFH-c4vcQVw
lightRag:https://mp.weixin.qq.com/s/JlRV1baeptqRS_nKOwuvtA
ZincSearch 是一个进行全文索引的搜索引擎:https://mp.weixin.qq.com/s/VGzqcM-3OTxNIi3KrBIRSA
Contextual RAG开源实现:https://mp.weixin.qq.com/s/E-kIExOtP1jZWtwhat0bdQ
KAG介绍:https://mp.weixin.qq.com/s/TqgGLlEYL5DqPEg6sB3tGA
Mistral 7B+Neo4j:构建知识图谱:https://mp.weixin.qq.com/s/7irzDGwMdvCaexcPE_iu8w
iText2KG:使用大型语言模型构建增量知识图谱:https://mp.weixin.qq.com/s/oiDffH1_0JiGpVGw83-guQ
"""


def read_md(input_md_file):
    with open(input_md_file, "r", encoding="utf-8") as f:
        content = f.read()

    return content


if __name__ == "__main__":
    # python test/rag/test_kg.py
    md_file = "no_git_oic/1db87098-3f2a-4516-8307-e0517a7ec98e/auto/1db87098-3f2a-4516-8307-e0517a7ec98e_tsr.md"
    content = read_md(md_file)
    output = chunk_by_LCEL(md_file, chunk_size=1500, chunk_overlap=300)
    for i, chunk in enumerate(output, start=1):
        print(f"the {i}:\n{chunk}")
