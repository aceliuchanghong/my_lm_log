"""
https://huggingface.co/docs/smolagents/index
https://github.com/huggingface/smolagents
"""

import os
from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool
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

if __name__ == "__main__":
    model_id = "deepseek-chat"
    api_base = "https://api.deepseek.com/v1"
    api_key = os.environ["DEEPSEEK_API_KEY"]
    """
    python test/usua2/test_smolagents.py
    """
    model = LiteLLMModel(model_id=model_id, api_base=api_base, api_key=api_key)
    # agent = CodeAgent(tools=[], model=model, add_base_tools=True)
    # agent.run(
    #     "Could you give me the 118th number in the Fibonacci sequence?",
    # )

    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
    print(agent.system_prompt_template)
    # agent.run(
    #     "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
    # )
