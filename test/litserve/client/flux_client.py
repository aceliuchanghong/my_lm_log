import requests
import os
from dotenv import load_dotenv
import logging
import time
import requests
import sys
import os
import json

os.environ["NUMEXPR_MAX_THREADS"] = "32"
load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from test.litserve.api.quick_ocr_api_server import base64_to_image


def generate_image(prompt, width=512, height=512, steps=4):
    url = f"http://localhost:{int(os.getenv('FLUX_PORT'))}/v1/images/generations"
    headers = {"Authorization": "Bearer torch-yzgjhdxfxfyzdjhljsjed5h"}
    save_dir = "no_git_oic/pics/flux"
    response = requests.post(
        url,
        json={
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": str(width) + "x" + str(height),
            "num_inference_steps": steps,
            "guidance_scale": 7,
        },
        headers=headers,
    )

    if response.status_code == 200:
        response_data = response.json()
        response_data = json.loads(response_data)
        image_base64 = response_data["data"][0]["url"]
        file_path = base64_to_image(image_base64, save_dir)
        return file_path
    else:
        print(
            f"Request failed with status code {response.status_code}: {response.text}"
        )
        return None


if __name__ == "__main__":
    start_time = time.time()
    prompt = "a pink robot sitting in a chair painting a picture on an easel of a futuristic cityscape, pop art"
    generate_image(prompt, 1024, 1024)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"耗时: {elapsed_time:.2f}秒")
    """
    export no_proxy="localhost,36.213.66.106,127.0.0.1,112.48.199.202,112.48.199.7"
    python test/litserve/client/flux_client.py
    """
