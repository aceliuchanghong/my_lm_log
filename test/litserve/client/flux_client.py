import requests
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import time
import requests
import sys


load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from test.litserve.api.quick_ocr_api_server import base64_to_image


logger = logging.getLogger(__name__)
url = f"http://localhost:{int(os.getenv('FLUX_PORT'))}/v1/images/generations"
headers = {"Authorization": "Bearer torch-yzgjhdxfxfyzdjhljsjed5h"}
prompt = "a pink robot sitting in a chair painting a picture on an easel of a futuristic cityscape, pop art"


start_time = time.time()
response = requests.post(
    url,
    json={
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "num_inference_steps": 8,
        "guidance_scale": 3.5,
    },
    headers=headers,
)
if response.status_code == 200:
    response_data = response.json()
    image_base64 = response_data["data"][0]["url"].split(",")[-1]
    print(f"{image_base64[:30]}")
    save_dir = "no_git_oic/pics"
    file_path = base64_to_image(image_base64, save_dir)
    logger.info(colored(f"saved in:{file_path}", "green"))

end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
"""
export no_proxy="localhost,36.213.66.106,127.0.0.1,1.12.251.149"
python test/litserve/client/flux_client.py
"""
