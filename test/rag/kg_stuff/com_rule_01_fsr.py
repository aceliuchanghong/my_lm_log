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


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/com_rule_01_fsr.py
    docx_path = "no_git_oic/com_rule_start"
    ASPECT = "公司培训管理制度"
    ai_tools = my_tools()

    docx_files = [
        os.path.join(docx_path, file)
        for file in os.listdir(docx_path)
        if file.endswith(".docx")
    ]
    for docs_file in docx_files:
        content_read = chunk_by_LCEL(
            read_docx(docs_file),
            chunk_size=int(os.getenv("chunk_size")),
            chunk_overlap=int(os.getenv("chunk_overlap")),
        )
        struct = ASPECT
        start_time = time.time()
        for contents in tqdm(content_read, desc="文件结构总结中..."):
            messages = [
                {
                    "role": "system",
                    "content": structure_fsr_prompt.format(
                        ASPECT=ASPECT, LEVEL="二级目录"
                    ),
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
                model=os.getenv("MODEL"),
                messages=messages,
                temperature=0.2,
            )
            logger.info(f"\n{response.choices[0].message.content}")
            struct = response.choices[0].message.content

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"结构树耗时: {elapsed_time:.2f}秒")
        break
