import os
from dotenv import load_dotenv
import logging
import time
import sys


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

from z_utils.get_ai_tools import my_tools


load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

start_time = time.time()


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    ai_tools = my_tools()
    start_time = time.time()

    response = ai_tools.llm.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[{"role": "user", "content": "1+1=?"}],
        temperature=0.2,
    )
    print(response.choices[0].message.content)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"耗时: {elapsed_time:.2f}秒")
