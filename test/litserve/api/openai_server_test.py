from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import litserve as ls
from litserve import LitAPI, LitServer
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import json

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LightRAGLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x + " from litserve"

    def decode_request(self, request):
        logger.info(colored(f"request:{request}", "green"))
        xx = [el.dict() for el in request.messages]
        logger.info(colored(f"xx1:{xx}", "green"))
        logger.info(colored(f"xx2:{xx[0]["content"]}", "green"))
        return json.dumps(xx[0]["content"]) + "mmm"

    def predict(self, x):
        yield self.model(x) + " 00"

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if (
            auth.scheme != "Bearer"
            or auth.credentials != "torch-bvisuwbksndclksjiocjwv742d"
        ):
            raise HTTPException(status_code=401, detail="Authorization Failed")

    def encode_response(self, output):
        x = ""
        for out in output:
            x += out
        data = {"xx": [1, 5, "00"], "output": x}
        yield {"role": "assistant", "content": str(data)}


if __name__ == "__main__":
    # python test/litserve/api/openai_server_test.py
    api = LightRAGLitAPI()
    server = LitServer(api, spec=ls.OpenAISpec())
    server.run(port=int(os.getenv("LIGHTRAG_PORT")))
