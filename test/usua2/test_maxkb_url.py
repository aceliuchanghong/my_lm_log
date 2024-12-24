import os
from dotenv import load_dotenv
import logging
from termcolor import colored
from openai import OpenAI
import time

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

prompt = "2023年营业利润?"
api_key = "application-50791c9c4f7a8d1f17fe084ca4dd3da9"
base_url = (
    "http://192.168.180.95:8080/api/application/c5f7b47c-b920-11ef-8d05-0242ac120003"
)
model = "gpt-3.5-turbo"
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
messages = [
    {"role": "user", "content": prompt},
]
start_time = time.time()
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.5,
)
logger.info(colored(f"original:{response.model_dump_json()}", "green"))
logger.info(
    colored(f"非流式测试llm ans:\n{response.choices[0].message.content}", "green")
)

stream_response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.5,
    stream=True,
)
for chunk in stream_response:
    logger.info(colored(f"chunk:{chunk.model_dump_json()}", "yellow"))

end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
# export no_proxy="localhost,127.0.0.1"
# python test/usua2/test_maxkb_url.py
