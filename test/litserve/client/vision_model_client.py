import requests
import time
import os
from dotenv import load_dotenv
import logging
import json

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
start_time = time.time()

ip = "127.0.0.1"
rule = {
    "entity_name": "条形码下方10位数字号码",
    "entity_format": "2100000010",
    "entity_regex_pattern": "[1-2][0-9]{9}",
}
rule_as_json = json.dumps(rule)
headers = {"Authorization": "Bearer torch-yzgjhdxfxfyzdjhljsjed5h"}
response = requests.post(
    f"http://{ip}:8110/v1/chat/completions",
    json={
        "messages": [
            # {"role": "user", "content": "条形码下方10位数字号码"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": rule_as_json},
                    # {"type": "text", "text": "条形码下方10位数字号码"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "no_git_oic/采购合同2.pdf_show_0.jpg"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg"
                        },
                    },
                ],
            },
        ],
    },
    headers=headers,
)
# print(f"{response.json()}")
print(f"{response.json()['choices'][0]['message']['content']}")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")

"""
export no_proxy="localhost,36.213.66.106,127.0.0.1"
python test/litserve/client/vision_model_client.py
{'result': '2100000010', 'entity_name': '条形码下方10位数字号码'}
"""
