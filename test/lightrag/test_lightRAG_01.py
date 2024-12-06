import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, ollama_embedding
from lightrag.utils import EmbeddingFunc
import sys
from dotenv import load_dotenv
import logging


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
WORKING_DIR = "./no_git_oic/dickens"
from test.rag.kg_stuff.com_rule_01_fsr import read_docx

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Qwen2.5",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="torch-elskenrgvoiserngviopsejrmoief",
        base_url="http://36.213.66.106:11433/v1/",
        **kwargs,
    )


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="bge-m3:latest", host="http://localhost:11434"
        ),
    ),
)

docx_path = "no_git_oic/com_rule_start/TE-MF-B004考勤管理制度V4.1-20231020.docx"
file_content = read_docx(docx_path)
rag.insert(file_content)

# # Perform naive search
# print(rag.query("病假相关条例?", param=QueryParam(mode="naive")))

# # Perform local search
# print(rag.query("病假相关条例?", param=QueryParam(mode="local")))

# # Perform global search
# print(rag.query("病假相关条例?", param=QueryParam(mode="global")))

# # Perform hybrid search
# print(rag.query("病假相关条例?", param=QueryParam(mode="hybrid")))
# python test/lightrag/test_lightRAG_01.py
# export no_proxy="localhost,36.213.66.106,127.0.0.1"
