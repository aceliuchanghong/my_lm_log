import os
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from dotenv import load_dotenv
from openai import OpenAI
from langchain_ollama import OllamaEmbeddings
import time

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


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python z_utils/get_ai_tools.py
    ai_tools = my_tools()
    start_time = time.time()

    response = ai_tools.llm.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[{"role": "user", "content": "1+1=?"}],
        temperature=0.2,
    )
    print(response.choices[0].message.content)
