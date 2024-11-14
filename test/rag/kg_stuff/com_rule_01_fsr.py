from docx import Document
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import sys
import time

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
from z_utils.get_ai_tools import my_tools
from z_utils.get_text_chunk import chunk_by_LCEL
from test.rag.kg_stuff.prompt import *


def read_docx(file_path):
    document = Document(file_path)
    content = ""
    for paragraph in document.paragraphs:
        content += paragraph.text + "\n"
    return content


def process_docx_files_fsr(
    docx_path, ai_tools, model, chunk_size, chunk_overlap, aspect, level="二级目录"
):
    """
    :param docx_path: 文档存放的文件路径
    :param aspect: 主题或结构树的方面
    :param ai_tools: 用于调用AI工具的工具类实例
    :param chunk_size: 每个chunk的大小
    :param chunk_overlap: 每个chunk的重叠部分
    :param model: 使用的AI模型名称
    """

    struct = aspect  # 初始化结构
    start_time = time.time()

    content_read = chunk_by_LCEL(
        read_docx(docx_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for contents in tqdm(content_read, desc="文件结构总结中..."):
        messages = [
            {
                "role": "system",
                "content": structure_fsr_prompt.format(ASPECT=aspect, LEVEL=level),
            },
            {
                "role": "user",
                "content": "".join(["当前已有的总结构树:\n", struct]),
            },
            {
                "role": "user",
                "content": "".join(["当前待总结页面的详细内容:\n", contents]),
            },
        ]
        response = ai_tools.llm.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        # logger.info(f"\n{response.choices[0].message.content}")
        struct = response.choices[0].message.content

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"\n{struct}")
    logger.info(f"结构树生成耗时: {elapsed_time:.2f}秒")
    return struct


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/com_rule_01_fsr.py
    docx_path = "no_git_oic/com_rule_start/TE-MF-B011培训管理制度V6.1-20230314.docx"
    ai_tools = my_tools()
    chunk_size = int(os.getenv("chunk_size"))
    chunk_overlap = int(os.getenv("chunk_overlap"))
    model = os.getenv("MODEL")
    aspect = "公司培训管理制度"
    level = "二级目录"

    structure = process_docx_files_fsr(
        docx_path, ai_tools, model, chunk_size, chunk_overlap, aspect, level
    )
