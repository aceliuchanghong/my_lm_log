import litserve as ls
from surya.model.detection.model import (
    load_model as load_det_model,
    load_processor as load_det_processor,
)
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import os
from dotenv import load_dotenv
import logging
import sys
import torch

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from test.ocr.test_surya import run_surya_ocr

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SuryaAPI(ls.LitAPI):
    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        det_model_path = os.getenv("SURYA_DET3_MODEL_PATH")
        rec_model_path = os.getenv("SURYA_REC2_MODEL_PATH")

        self.rec_processor = load_rec_processor()
        self.det_model = load_det_model(det_model_path)
        self.det_processor = load_det_processor(det_model_path)
        self.rec_model = load_rec_model(rec_model_path)

    def predict(self, inputs):
        try:
            ocr_results = {}
            images_path = inputs["images_path"]
            for image in images_path:
                ocr_result = run_surya_ocr(
                    image,
                    self.det_model,
                    self.det_processor,
                    self.rec_model,
                    self.rec_processor,
                )
                ocr_results[image] = ocr_result
            return ocr_results
        except Exception as e:
            logger.error(f"error:{e}")
        finally:
            self.clean_memory(self.device)

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    # python test/litserve/api/surya_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # nohup python test/litserve/api/surya_server.py> no_git_oic/surya_server.log &
    api = SuryaAPI()
    server = ls.LitServer(api, devices=[1])
    server.run(port=int(os.getenv("SURYA_PORT")))
