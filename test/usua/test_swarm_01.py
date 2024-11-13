import os
from dotenv import load_dotenv
import logging
from swarm import Swarm, Agent
import sys


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
from z_utils.get_ai_tools import my_tools

ai_tools = my_tools()
client = Swarm(client=ai_tools.llm)


def get_weather(location) -> str:
    return "{'temp':67, 'unit':'F'}"


# export no_proxy="localhost,112.48.199.202,127.0.0.1"
# python test/usua/test_swarm_01.py
agent = Agent(
    name="Agent",
    model=os.getenv("MODEL"),
    instructions="1.every time i ask,temp add 1 in the basic of last time",
    functions=[get_weather],
)

messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.run(agent=agent, messages=messages)
print(response.messages[-1]["content"])
response = client.run(agent=agent, messages=messages)
print(response.messages[-1]["content"])
