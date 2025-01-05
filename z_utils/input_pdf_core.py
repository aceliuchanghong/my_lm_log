import os
import sys
from pdf2image import convert_from_path
from dotenv import load_dotenv
from tqdm import tqdm
import logging
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
)
from z_utils.check_db import excute_sqlite_sql
from z_utils.sql_sentence import select_rule_sql


def process_pdf(pdf_file_path):
    # apt install poppler-utils
    pdf2img_output_path = os.path.join(
        os.getenv("upload_file_save_path", "./upload_files"),
        "entity_extract",
        os.path.basename(pdf_file_path).split(".")[0],
    )
    os.makedirs(pdf2img_output_path, exist_ok=True)
    images = convert_from_path(pdf_file_path)
    image_paths = []
    for i, image in enumerate(tqdm(images, desc="Converting PDF to PNG images")):
        image_path = os.path.join(pdf2img_output_path, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


def process_file(file_original):
    if file_original is None:
        return ["z_using_files/pics/ell-wide-dark.png"], []
    if file_original.endswith(".pdf"):
        cut_pics = process_pdf(file_original)
        return cut_pics, cut_pics
    else:
        return [file_original], [file_original]


def get_rule_list(rule_name):
    entity_tuple_list = excute_sqlite_sql(select_rule_sql, (rule_name,), False)
    tasks = []
    for i, entity in enumerate(entity_tuple_list):
        task = {
            "entity_name": entity[0],
            "entity_format": entity[1],
            "entity_regex_pattern": entity[2],
        }
        tasks.append(task)
    return tasks
