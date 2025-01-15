import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
import glob
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

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
from test.usua3.try_poem import save_new_json


def get_poet_song_json_files(directory, pattern):
    """
    获取指定目录下所有符合 pattern 模式的文件路径。

    参数:
        directory (str): 目标目录路径。
        pattern (str): 文件名匹配模式（例如 'poet.song.*.json'）。

    返回:
        list: 包含所有匹配文件路径的列表。
    """
    json_files = glob.glob(os.path.join(directory, pattern))

    return json_files


if __name__ == "__main__":
    """
    export no_proxy="localhost,127.0.0.1"
    python test/usua3/start_poem_tang.py
    nohup python test/usua3/start_poem_tang.py > no_git_oic/start_poem_tang.log &
    """
    directory = "../chinese-poetry/全唐诗/"
    pattern = "poet.song.*.json"
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMB_MODEL", "bge-m3"), base_url=os.getenv("EMB_BASE_URL")
    )
    client = OpenAI(
        api_key=os.getenv("API_KEY"), base_url=os.getenv("OLLAMA_CHAT_BASE_URL")
    )

    json_files = get_poet_song_json_files(directory, pattern)
    for json_file in tqdm(json_files, desc="Processing json file"):
        deal_nums = save_new_json(json_file, client, embeddings)
        logger.debug(colored(f"deal_nums:{deal_nums}", "green"))
