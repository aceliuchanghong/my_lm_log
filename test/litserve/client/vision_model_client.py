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

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:8110/predict",
    json={
        "images_path": [
            "./upload_files/images/发票签收单2.pdf_show_0.jpg",
            "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg",
        ],
        "rule": {
            "entity_name": "10位数条形码号码",
            "entity_format": "2100000010",
            "entity_regex_pattern": "[1-2][0-9]{9}",
        },
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")

"""
Status: 200
Response:
 {"result":"2200000003","entity_name":"10位数条形码号码"}
"""
