from pymilvus import MilvusClient
import os
from dotenv import load_dotenv
import logging
import time


load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # export no_proxy="localhost,127.0.0.1"
    start_time = time.time()

    logger.info(f"connect milvus")
    client = MilvusClient(uri=os.getenv("MILVUS_URI"))

    has = client.has_collection("hello_milvus")
    logger.info(f"Does collection hello_milvus exist in Milvus?: {has}")

    with open("z_using_files/txt/太白金星有点烦.txt", "r", encoding="gbk") as f:
        cont = f.read()
        logger.debug(f"{cont[:100]}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"耗时: {elapsed_time:.2f}秒")
