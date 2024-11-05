from litellm import completion
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
response = completion(
    model="ollama/" + os.getenv("MODEL"),
    api_base=os.getenv("EMB_BASE_URL"),
    messages=[{"content": "respond in 20 words. who are you?", "role": "user"}],
    stream=True,
)
# python test/llm/test_litellm.py
print(colored(response, "yellow"))
for chunk in response:
    print(chunk["choices"][0]["delta"])
