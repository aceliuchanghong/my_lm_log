import os
from dotenv import load_dotenv
import logging
import sys
from termcolor import colored

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
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from test.lightrag.prompt import PROMPTS

kw_prompt_temp = PROMPTS["keywords_extraction"]
query = "型号为GH-09的零件的相关性质信息"
kw_prompt = kw_prompt_temp.format(query=query)
print(f"{kw_prompt}")
# python test/lightrag/test_lightRAG_03.py
