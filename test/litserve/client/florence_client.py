import requests
import time
import os
from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
start_time = time.time()


url = "http://127.0.0.1:8117/predict"
data = {
    "images": [
        "/mnt/data/llch/fluxgym/pics/000.png",
        "/mnt/data/llch/fluxgym/pics/6谷雨.png",
        "/mnt/data/llch/fluxgym/pics/1立春.png",
    ],
}

response = requests.post(url, json=data)

print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
# export no_proxy="localhost,112.48.199.202,127.0.0.1"
# python test/litserve/client/florence_client.py
