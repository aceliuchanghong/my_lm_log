import requests
import json
from termcolor import colored
import os
from dotenv import load_dotenv
import logging
import time

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
API_KEY = "xx"
SECRET_KEY = "xx"


def main(input_s):

    url = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token="
        + get_access_token()
    )

    payload = json.dumps(
        {
            "messages": [
                {"role": "user", "content": input_s},
            ],
            "temperature": 0.95,
            "top_p": 0.8,
            "penalty_score": 1,
            "enable_system_memory": False,
            "disable_search": False,
            "enable_citation": False,
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    return json.loads(response.text)["result"]


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY,
    }
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/usua/test_wenxinyiyan.py
    start_time = time.time()
    input_s = "今天是哪天?今天上海天气如何?"
    x = main(input_s)
    logger.info(colored(f"\n{x}", "green"))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"耗时: {elapsed_time:.2f}秒")
