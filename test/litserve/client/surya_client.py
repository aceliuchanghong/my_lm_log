import requests
import os
from dotenv import load_dotenv
import logging
import json
import concurrent.futures

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
import time

start_time = time.time()

ip = "127.0.0.1"
images_paths = [
    "no_git_oic/Snipaste_2024-10-14_13-41-02.png",
    "no_git_oic/page_1.png",
]
max_workers = 8


def send_request():
    response = requests.post(
        f"http://{ip}:{os.getenv('SURYA_PORT')}/predict",
        json={"images_path": images_paths},
    )
    return response


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(send_request) for _ in range(max_workers)]
    for future in concurrent.futures.as_completed(futures):
        response = future.result()
        print(
            f"response:{response}\nStatus: {response.status_code}\nResponse:\n {response.text}\n dict:{json.loads(response.text)['output']}"
        )
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
# python test/litserve/client/surya_client.py
# export no_proxy="localhost,112.48.199.202,127.0.0.1"
