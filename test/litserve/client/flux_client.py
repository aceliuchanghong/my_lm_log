import requests
import json
import os
from dotenv import load_dotenv
import logging
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
url = f"http://localhost:{int(os.getenv('FLUX_PORT'))}/predict"
headers = {"Authorization": "Bearer torch-yzgjhdxfxfyzdjhljsjed5h"}
prompt = "a robot sitting in a chair painting a picture on an easel of a futuristic cityscape, pop art"
response = requests.post(
    url,
    json={
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    },
    headers=headers,
)
with open("generated_image.png", "wb") as f:
    f.write(response.content)
print("Image generated and saved as generated_image.png")
