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
from test.rag.kg_stuff.com_rule_01_fsr import read_docx


if __name__ == "__main__":
    # python test/lightrag/test_lightRAG_02.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    docx_path1 = "no_git_oic/com_rule_start/TE-MF-B004考勤管理制度V4.1-20231020.docx"
    file_content1 = read_docx(docx_path1)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    x = PROMPTS["summarize_entity_descriptions"].format(
        entity_name="00", description_list="11"
    )
    logger.info(colored(f"{x}", "green"))
    hint_prompt = entity_extract_prompt.format(**context_base, input_text=file_content1)
    logger.info(colored(f"{hint_prompt}", "green"))
