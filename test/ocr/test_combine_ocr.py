import os
from dotenv import load_dotenv
import logging
from PIL import Image

# from surya.model.detection.model import (
#     load_model as load_det_model,
#     load_processor as load_det_processor,
# )
# from surya.model.recognition.model import load_model as load_rec_model
# from surya.model.recognition.processor import load_processor as load_rec_processor
# from rapidocr_onnxruntime import RapidOCR
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

# from test.ocr.test_surya import polygon_to_markdown, run_surya_ocr


load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
import time


class NewTextLine:
    def __init__(self, polygon, confidence, text, bbox):
        self.polygon = polygon
        self.confidence = confidence
        self.text = text
        self.bbox = bbox


def create_textline_from_data(data):
    polygon = data[0]  # 取出polygon坐标
    text = data[1]  # 取出OCR识别的文本
    confidence = data[2]  # 取出置信度

    # 根据polygon计算出bbox: [x_min, y_min, x_max, y_max]
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]

    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    return NewTextLine(polygon=polygon, confidence=confidence, text=text, bbox=bbox)


if __name__ == "__main__":
    IMAGE_PATH = "no_git_oic/page_1.png"
    IMAGE_PATH = "no_git_oic/Snipaste_2024-10-14_13-41-02.png"

    det_model_path = os.getenv("SURYA_DET3_MODEL_PATH")
    rec_model_path = os.getenv("SURYA_REC2_MODEL_PATH")
    layout_model_path = os.getenv("SURYA_LAYOUT4_MODEL_PATH")
    order_model_path = os.getenv("SURYA_ORDER_MODEL_PATH")
    table_rec_model_path = os.getenv("SURYA_TABLEREC_MODEL_PATH")

    start_time = time.time()

    rec_processor = load_rec_processor()
    det_model = load_det_model(det_model_path)
    det_processor = load_det_processor(det_model_path)
    rec_model = load_rec_model(rec_model_path)
    ocr_engine = RapidOCR()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"ocr模型加载耗时: {elapsed_time:.2f}秒")

    IMAGE_PATH2 = Image.open(IMAGE_PATH)
    rapid_ocr_result, _ = ocr_engine(IMAGE_PATH2)
    text_lines = []
    for line in rapid_ocr_result:
        text_line = create_textline_from_data(line)
        text_lines.append(text_line)
    markdown0 = polygon_to_markdown(text_lines)
    markdown1 = markdown0.splitlines()[:14]

    rapid_ocr_markdown = "\n".join([text for text in markdown1 if len(text) > 0])
    print(f"{rapid_ocr_markdown}")

    ocr_result = run_surya_ocr(
        IMAGE_PATH, det_model, det_processor, rec_model, rec_processor
    )
