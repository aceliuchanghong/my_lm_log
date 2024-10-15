import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
import time

start_time = time.time()

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:{int(os.getenv('QWEN2_VL_PORT'))}/predict",
    json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "no_git_oic/page_3.png",
                        "max_pixels": 512 * 512,
                    },
                    # {
                    #     "type": "image",
                    #     "image": "no_git_oic/企业微信截图_17288805261441.png",
                    # },
                    # {
                    #     "type": "image",
                    #     "image": "no_git_oic/Snipaste_2024-10-14_13-41-02.png",
                    # },
                    {
                        "type": "text",
                        # "text": "提取立项必要性分析-背景,市场前景,意义,以json格式返回",
                        "text": "还原文档,使用markdown格式输出",
                        # "text": "SOB号码多少?格式是SOB+6位数字+横线+数字",
                    },
                ],
            }
        ],
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
