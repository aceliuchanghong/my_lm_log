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


url = "http://127.0.0.1:8927/predict"
data = {
    "image_path": "no_git_oic/采购合同4.pdf_show_0.jpg",
    # "text_input": "提取SOB号码,它的可能结果案例:SOB20..-..它的可能结果正则:S[Oo0][BA](\d{6}|\d{8})-\d{5}",
    "text_input": "提取图片中kv,json格式返回",
    # "text_input": "将图片还原回markdown格式",
}

response = requests.post(url, json=data)

print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
