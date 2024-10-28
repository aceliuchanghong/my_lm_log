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
    f"http://{ip}:{os.getenv('SURYA_PORT')}/predict",
    json={
        "images_path": [
            "no_git_oic/Snipaste_2024-10-14_13-41-02.png",
            "no_git_oic/page_1.png",
        ],
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
# python test/litserve/client/surya_client.py
# export no_proxy="localhost,112.48.199.202,127.0.0.1"
