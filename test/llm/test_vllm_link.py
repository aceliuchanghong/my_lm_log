import os
from dotenv import load_dotenv
import logging
from termcolor import colored
from openai import OpenAI
import concurrent.futures

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def send_request(client, messages):
    return client.chat.completions.create(
        model="Qwen2.5",
        messages=messages,
        temperature=0.5,
    )


if __name__ == "__main__":
    # export no_proxy="localhost,36.213.66.106,127.0.0.1"
    # python test/llm/test_vllm_link.py
    client = OpenAI(
        api_key="torch-elskenrgvoiserngviopsejrmoief",
        base_url="http://36.213.66.106:11433/v1",
    )
    messages = [
        {"role": "system", "content": "你叫火炬AI助手"},
        {"role": "user", "content": "你是谁?帮我写一首中文七言绝句诗"},
    ]
    messages_list = [messages] * 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(
            executor.map(lambda msg: send_request(client, msg), messages_list)
        )
        logger.info(colored(f"{results}", "green"))
