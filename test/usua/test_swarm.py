from swarm import Swarm, Agent
import os
from dotenv import load_dotenv
import logging
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


def transfer_to_agent_b():
    return agent_b


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/usua/test_swarm.py
    ai_tools = my_tools()
    client = Swarm(client=ai_tools.llm)

    agent_a = Agent(
        name="Agent A",
        model=os.getenv("SMALL_MODEL"),
        instructions="You are a helpful agent.",
        functions=[transfer_to_agent_b],
    )

    agent_b = Agent(
        name="Agent B",
        model=os.getenv("SMALL_MODEL"),
        instructions="1.Respond in Chinese. 2.不管接收到什么用户输入,都返回:'喵喵喵'",
    )

    response = client.run(
        agent=agent_a,
        messages=[
            {"role": "user", "content": "I want to talk to agent B."},
        ],
    )
    print(response.messages[-1]["content"])
