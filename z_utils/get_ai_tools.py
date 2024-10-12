import os
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from dotenv import load_dotenv
from openai import OpenAI
from langchain_ollama import OllamaEmbeddings


load_dotenv()


class my_tools:
    def __init__(self) -> None:
        self.table_engine = RapidTable(
            model_path=os.getenv("rapidocr_table_engine_model_path")
        )
        self.ocr_engine = RapidOCR()
        self.llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("EMB_MODEL"), base_url=os.getenv("EMB_BASE_URL")
        )
