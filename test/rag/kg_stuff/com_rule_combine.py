import os
from dotenv import load_dotenv
import logging
import sys
import time
import argparse
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
from test.rag.kg_stuff.com_rule_01_fsr import process_docx_files_fsr
from test.rag.kg_stuff.com_rule_02_kg import process_docx_files_kg
from z_utils.get_ai_tools import my_tools


if __name__ == "__main__":
    """
    export no_proxy="localhost,112.48.199.202,127.0.0.1"
    python test/rag/kg_stuff/com_rule_combine.py
    python test/rag/kg_stuff/com_rule_combine.py --docx_path "no_git_oic/com_rule_start/TE-MF-B011培训管理制度V6.1-20230314.docx" --aspect "公司培训管理制度"
    python test/rag/kg_stuff/com_rule_combine.py \
        --docx_path "no_git_oic/1db87098-3f2a-4516-8307-e0517a7ec98e/auto/1db87098-3f2a-4516-8307-e0517a7ec98e_tsr.md" \
        --aspect "电容器检验记录"
    """
    parser = argparse.ArgumentParser(description="文件处理")
    parser.add_argument(
        "--docx_path",
        default="no_git_oic/com_rule_start/TE-MF-B004考勤管理制度V4.1-20231020.docx",
        help="文档路径，支持.docx和.md格式",
    )
    parser.add_argument("--aspect", default="考勤管理制度", help="分析的具体方面")
    args = parser.parse_args()

    ai_tools = my_tools()
    chunk_size = int(os.getenv("chunk_size"))
    chunk_overlap = int(os.getenv("chunk_overlap"))
    model = os.getenv("MODEL")
    docx_path = args.docx_path
    aspect = args.aspect

    start_time = time.time()
    structure_list = process_docx_files_fsr(
        docx_path, ai_tools, model, chunk_size, chunk_overlap, aspect
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"structure_list耗时: {elapsed_time:.2f}秒")

    start_time = time.time()
    kg_list = process_docx_files_kg(
        docx_path, ai_tools, model, chunk_size, chunk_overlap, structure_list, aspect
    )
    logger.info(f"{kg_list}")
    elapsed_time = time.time() - start_time
    logger.info(f"kg生成耗时: {elapsed_time:.2f}秒")
    with open("no_git_oic/fsr_kg.txt", "w") as f:
        json.dump(kg_list, f, ensure_ascii=False, indent=4)
