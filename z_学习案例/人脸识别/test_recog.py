import os
from dotenv import load_dotenv
import logging
from termcolor import colored
from deepface import DeepFace

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# python z_学习案例/人脸识别/test_recog.py
result = DeepFace.verify(
    img1_path="no_git_oic/torch_pics/lch_1.png",
    img2_path="no_git_oic/torch_pics/zzw_1.jpg",
)
logger.info(colored(f"{result}", "green"))
