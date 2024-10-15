from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
import logging
from langchain_community.document_loaders import PyPDFLoader

# https://github.com/AuvaLab/itext2kg/blob/main/examples/different_llm_models.ipynb
# export no_proxy="localhost,112.48.199.202,127.0.0.1"

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0.1,
)
embeddings = OllamaEmbeddings(
    model=os.getenv("EMB_MODEL"), base_url=os.getenv("EMB_BASE_URL")
)


loader = PyPDFLoader(f"no_git_oic/页面提取自－NPD2317设计开发记录.pdf")
pages = loader.load_and_split()
print(pages)
