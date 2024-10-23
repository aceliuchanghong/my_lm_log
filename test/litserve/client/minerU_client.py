import json
import pymupdf
import requests
import numpy as np
from loguru import logger
from joblib import Parallel, delayed
import os
from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


def to_pdf(file_path):
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            pdf_bytes = f.tobytes()
        else:
            pdf_bytes = f.convert_to_pdf()
        return pdf_bytes


def do_parse(
    file_path,
    url=f"http://127.0.0.1:{int(os.getenv('MINERU_SERVER_PORT'))}/predict",
    **kwargs,
):
    try:
        kwargs.setdefault("parse_method", "auto")
        kwargs.setdefault("debug_able", False)

        response = requests.post(
            url, data={"kwargs": json.dumps(kwargs)}, files={"file": to_pdf(file_path)}
        )

        if response.status_code == 200:
            output = response.json()
            output["file_path"] = file_path
            return output
        else:
            raise Exception(response.text)
    except Exception as e:
        logger.error(f"File: {file_path} - Info: {e}")


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/litserve/client/minerU_client.py
    files = ["no_git_oic/页面提取自－NPD2317设计开发记录.pdf"]
    n_jobs = np.clip(len(files), 1, 4)
    results = Parallel(n_jobs, prefer="threads", verbose=10)(
        delayed(do_parse)(p) for p in files
    )
    print(results)
