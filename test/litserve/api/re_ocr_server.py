import litserve as ls
from openai import OpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from rapidocr_onnxruntime import RapidOCR
import logging
import sys
import os
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
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from test.litserve.api.quick_ocr_api_server import extract_entity, get_local_images
from test.ocr.test_surya import polygon_to_markdown
from test.ocr.test_combine_ocr import create_textline_from_data
from z_utils.get_text_chunk import chunk_by_LCEL


class ReOcrLitAPI(ls.LitAPI):
    def re_ocr_image(self, local_image):
        single_result = {}
        single_result["file_name"] = os.path.basename(local_image)

        rapid_ocr_result, _ = self.ocr_engine(local_image)
        text_lines = []
        for line in rapid_ocr_result:
            text_line = create_textline_from_data(line)
            text_lines.append(text_line)
        markdown0 = polygon_to_markdown(text_lines)
        markdown1 = markdown0.splitlines()
        rapid_ocr_markdown = "\n".join([text for text in markdown1 if len(text) > 0])
        logger.debug(f"rapid_ocr_markdown:\n{rapid_ocr_markdown}")
        return rapid_ocr_markdown

    def setup(self, device):
        self.ocr_engine = RapidOCR()
        self.llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("EMB_MODEL"), base_url=os.getenv("EMB_BASE_URL")
        )

    def decode_request(self, request):
        logger.info(f"received requests: {request}")
        images_path = request["images_path"]
        rule = request["rule"]
        local_images_path = get_local_images(images_path)

        # logger.info(f"local_images_path:\n{local_images_path}")
        return local_images_path, rule

    def predict(self, inputs):
        local_images_path, rule = inputs
        ocr_result = ""
        for local_image in local_images_path:
            ocr_result += self.re_ocr_image(local_image) + "\n"
            # 删除文件
            try:
                save_dir = os.path.join(os.getenv("upload_file_save_path"), "images")
                rotate_path = os.path.join(
                    os.getenv("upload_file_save_path"), "rotate_pics"
                )
                upload_image = os.path.join(save_dir, os.path.basename(local_image))
                rotate_image = os.path.join(rotate_path, os.path.basename(local_image))
                os.remove(upload_image)
                os.remove(rotate_image)
            except Exception as e:
                pass
        logger.info(f"ocr_result:\n{ocr_result}")

        retriever = None
        if len(ocr_result) > int(os.getenv("long_ocr_result")):
            text_doc = chunk_by_LCEL(ocr_result)
            vectorstore = InMemoryVectorStore.from_texts(
                text_doc,
                embedding=self.embeddings,
            )
            retriever = vectorstore.as_retriever()

        ans = extract_entity(self.llm, rule, ocr_result, retriever)

        return ans

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    # python test/litserve/api/re_ocr_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    api = ReOcrLitAPI()
    server = ls.LitServer(api, accelerator="auto", devices=1)
    server.run(port=int(os.getenv("MOLMO_PORT")))
