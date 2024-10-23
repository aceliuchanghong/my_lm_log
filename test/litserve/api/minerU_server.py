import torch
import filetype
import json, uuid
import litserve as ls
from unittest.mock import patch
from fastapi import HTTPException
from magic_pdf.tools.common import do_parse
from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
import os
from dotenv import load_dotenv
import logging


load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
"""
只适配3.10,md,且kit1.0模型没适配
pip install -U litserve python-multipart filetype
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
{
    "bucket_info":{
        "bucket-name-1":["ak", "sk", "endpoint"],
        "bucket-name-2":["ak", "sk", "endpoint"]
    },
    "models-dir":"/mnt/data/llch/PDF-Extract-Kit/models",
    "device-mode":"cuda",
    "table-config": {
        "model": "TableMaster",
        "is_table_recog_enable": false,
        "max_time": 400
    }
}
"""


class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir="/tmp"):
        self.output_dir = output_dir

    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        with patch(
            "magic_pdf.model.doc_analyze_by_custom_model.get_device"
        ) as mock_obj:
            mock_obj.return_value = device
            self.models = ModelSingleton()
            model_manager = self.models
            model_manager.get_model(True, False)
            model_manager.get_model(False, False)
            mock_obj.assert_called()
            print(f"Model initialization complete!")

    def decode_request(self, request):
        file = request["file"].file.read()
        kwargs = json.loads(request["kwargs"])
        assert filetype.guess_mime(file) == "application/pdf"
        return file, kwargs

    def predict(self, inputs):
        try:
            pdf_name = str(uuid.uuid4())
            do_parse(self.output_dir, pdf_name, inputs[0], [], **inputs[1])
            return pdf_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        finally:
            self.clean_memory(self.device)

    def encode_response(self, response):
        return {"output_dir": response}


if __name__ == "__main__":
    # python test/litserve/api/minerU_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    server = ls.LitServer(MinerUAPI(), accelerator="gpu", devices=[3])
    server.run(port=int(os.getenv("MINERU_SERVER_PORT")))
