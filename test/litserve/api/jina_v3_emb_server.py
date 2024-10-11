# https://huggingface.co/jinaai/jina-embeddings-v3
# V100跑不起来
import os
import time
import litserve as ls
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


class JinaEmbAPI(ls.LitAPI):
    def setup(self, device):
        self.emb_model = SentenceTransformer(
            os.getenv("JINA_MODEL_PATH"), trust_remote_code=True, device=device
        )

    def decode_request(self, request):
        return request["input"]

    def predict(self, query):
        start_time = time.time()
        result = self.emb_model.encode([query])
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"EMB耗时: {elapsed_time:.2f}秒")
        return result

    def encode_response(self, output):
        return {"embedding": output[0].tolist()}


if __name__ == "__main__":
    # python test/litserve/api/jina_v3_emb_server.py
    # export no_proxy="localhost,127.0.0.1"
    api = JinaEmbAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=1)
    server.run(port=int(os.getenv("JINA_PORT")))
