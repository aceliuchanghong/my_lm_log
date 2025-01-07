import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import litserve as ls
import time

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
from test.ocr.test_surya import polygon_to_markdown
from test.ocr.test_combine_ocr import create_textline_from_data
from test.litserve.api.quick_ocr_api_server import get_local_images


class OcrAPI(ls.LitAPI):
    def setup(self, device):
        self.table_engine = RapidTable(
            model_path=os.getenv(
                "rapidocr_table_engine_model_path",
                "./test/ocr/ch_ppstructure_mobile_v2_SLANet.onnx",
            )
        )
        self.ocr_engine = RapidOCR()

    def decode_request(self, request):
        logger.debug(f"received request:\n{request}")
        images_path = request["images_path"]
        mode = request["mode"]
        local_images_path = get_local_images(images_path)

        return local_images_path, mode

    def predict(self, inputs):
        local_images_path, mode = inputs
        start_time = time.time()
        result = {}
        for local_image in local_images_path:
            rapid_ocr_result, _ = self.ocr_engine(local_image)
            if mode == "normal":
                result[local_image] = rapid_ocr_result
            elif mode == "markdown":
                text_lines = []
                if rapid_ocr_result is None:
                    rapid_ocr_result = []
                for line in rapid_ocr_result:
                    text_line = create_textline_from_data(line)
                    text_lines.append(text_line)
                markdown0 = polygon_to_markdown(text_lines)
                markdown1 = markdown0.splitlines()
                rapid_ocr_markdown = "\n".join(
                    [text for text in markdown1 if len(text) > 0]
                )
                result[local_image] = rapid_ocr_markdown
            elif mode == "html":
                ss = ""
                if rapid_ocr_result is not None:
                    for buck in rapid_ocr_result:
                        ss += buck[1] + " "
                    table_html_str = self.table_engine(local_image, rapid_ocr_result)[0]
                    ss = table_html_str.replace("<html><body>", "").replace(
                        "</body></html>", ""
                    )
                result[local_image] = ss
            else:
                raise ValueError(
                    f"mode: {mode} is not supported,only:normal,markdown,html"
                )
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"OCR耗时: {elapsed_time:.2f}秒")
        return result

    def encode_response(self, output):
        logger.debug(colored(f"output:\n{output}", "green"))
        return output

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if (
            auth.scheme != "Bearer"
            or auth.credentials != "torch-yzgjsvedioyzdjhljsjed5h"
        ):
            raise HTTPException(status_code=401, detail="Authorization Failed")


if __name__ == "__main__":
    """
    export no_proxy="localhost,127.0.0.1,1.12.251.149"
    python test/litserve/api/ocr_server.py
    nohup python test/litserve/api/ocr_server.py > no_git_oic/ocr_server.log &
    """
    api = OcrAPI()
    server = ls.LitServer(api, devices=1, track_requests=True)
    server.run(port=int(os.getenv("OCR_PORT", 8124)))
