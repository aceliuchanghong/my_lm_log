"""
Loaded detection model vikp/surya_det3 on device cuda with dtype torch.float16
Loaded recognition model vikp/surya_rec2 on device cuda with dtype torch.float16
Loaded detection model vikp/surya_layout3 on device cuda with dtype torch.float16
Loaded reading order model vikp/surya_order on device cuda with dtype torch.float16
Loaded recognition model vikp/surya_tablerec on device cuda with dtype torch.float16

class TextLine(PolygonBox):
    text: str
    confidence: Optional[float] = None
    
class OCRResult(BaseModel):
    text_lines: List[TextLine]
    languages: List[str] | None = None
    image_bbox: List[float]
"""

import os
from dotenv import load_dotenv
import logging
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import (
    load_model as load_det_model,
    load_processor as load_det_processor,
)
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.ordering import batch_ordering
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.ordering.model import load_model as load_order_model
from tabled.extract import extract_tables
from tabled.fileinput import load_pdfs_images
from surya.model.table_rec.model import load_model as load_table_rec_model
from surya.model.table_rec.processor import load_processor as load_table_rec_processor
from surya.settings import settings

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
import time


def polygon_to_markdown(text_lines):
    # 将 TextLine 对象列表转换为 Markdown 格式
    markdown_text = ""
    previous_y = -1

    # 对文本框按y坐标进行排序
    sorted_lines = sorted(text_lines, key=lambda line: line.bbox[1])

    for line in sorted_lines:
        # 获取当前文本框的 y 坐标
        current_y = line.bbox[1]

        # 根据 y 坐标检测是否需要换段落
        if previous_y != -1 and abs(current_y - previous_y) > 20:
            markdown_text += "\n\n"  # 大间距时换段落
        else:
            markdown_text += " "  # 否则按空格分隔

        markdown_text += line.text
        previous_y = current_y

    return markdown_text.strip()


def run_surya_ocr(IMAGE_PATH, det_model, det_processor, rec_model, rec_processor):
    """
    ocr 结果
    """
    start_time = time.time()
    image = Image.open(IMAGE_PATH)
    langs = ["zh", "en"]
    predictions = run_ocr(
        [image], [langs], det_model, det_processor, rec_model, rec_processor
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"surya_ocr耗时: {elapsed_time:.2f}秒")

    for text_line in predictions[0].text_lines:
        logger.debug(f"text_line:{text_line}")
        logger.debug(f"text:{text_line.text}")
        logger.debug(f"polygon:{text_line.polygon}")
        logger.debug(f"bbox:{text_line.bbox}")

    markdown_predictions0 = polygon_to_markdown(predictions[0].text_lines)
    markdown_predictions1 = markdown_predictions0.splitlines()
    markdown_predictions = "\n".join(
        [text for text in markdown_predictions1 if len(text) > 0]
    )
    logger.debug(f"markdown:\n{markdown_predictions}")
    return markdown_predictions


def run_surya_batch_text_detection(IMAGE_PATH, det_model, det_processor):
    """
    Text line detection 文本行检测
    给出一个json,其中包含检测到的bbox
    """

    start_time = time.time()

    image = Image.open(IMAGE_PATH)
    text_detection = batch_text_detection([image], det_model, det_processor)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"surya_text_detection耗时: {elapsed_time:.2f}秒")
    return text_detection


def run_surya_batch_layout_detection(
    IMAGE_PATH, layout_model, layout_processor, det_model, det_processor
):
    """
    Layout analysis 版面分析
    布局检测
    """
    start_time = time.time()

    image = Image.open(IMAGE_PATH)

    line_predictions = batch_text_detection([image], det_model, det_processor)
    layout_predictions = batch_layout_detection(
        [image], layout_model, layout_processor, line_predictions
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"surya_layout_detection耗时: {elapsed_time:.2f}秒")
    return layout_predictions


def run_surya_order_detection(IMAGE_PATH, order_model, order_processor):
    """
    Reading order box顺序
    """
    start_time = time.time()

    image = Image.open(IMAGE_PATH)
    # bboxes should be a list of lists with layout bboxes for the image in [x1,y1,x2,y2] format
    # You can get this from the layout model, see above for usage
    bboxes = [[1588, 133, 1742, 182]]

    order_predictions = batch_ordering([image], [bboxes], order_model, order_processor)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"surya_order_detection耗时: {elapsed_time:.2f}秒")
    return order_predictions


def run_surya_table_detection(
    PDF_PATH,
    det_model,
    det_processor,
    layout_model,
    layout_processor,
    table_rec_model,
    table_rec_processor,
    rec_model,
    rec_processor,
):
    """
    table rec
    """

    new_det_model = det_model, det_processor, layout_model, layout_processor
    new_rec_models = table_rec_model, table_rec_processor, rec_model, rec_processor

    start_time = time.time()

    images, highres_images, names, text_lines = load_pdfs_images(PDF_PATH)
    table_detection = extract_tables(
        images, highres_images, text_lines, new_det_model, new_rec_models
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"surya_table_detection耗时: {elapsed_time:.2f}秒")
    return table_detection


if __name__ == "__main__":
    IMAGE_PATH = "no_git_oic/page_1.png"
    # IMAGE_PATH = "no_git_oic/发票签收单2.pdf_show_0.jpg"
    PDF_PATH = "no_git_oic/页面提取自－NPD2317设计开发记录.pdf"

    det_model_path = os.getenv("SURYA_DET3_MODEL_PATH")
    rec_model_path = os.getenv("SURYA_REC2_MODEL_PATH")
    layout_model_path = os.getenv("SURYA_LAYOUT4_MODEL_PATH")
    order_model_path = os.getenv("SURYA_ORDER_MODEL_PATH")
    table_rec_model_path = os.getenv("SURYA_TABLEREC_MODEL_PATH")

    start_time = time.time()
    settings.TORCH_DEVICE = "cuda:2"

    rec_processor = load_rec_processor()
    det_model = load_det_model(det_model_path)
    det_processor = load_det_processor(det_model_path)
    rec_model = load_rec_model(rec_model_path)

    layout_model = load_det_model(layout_model_path)
    layout_processor = load_det_processor(layout_model_path)

    order_model = load_order_processor(order_model_path)
    order_processor = load_order_model(order_model_path)

    table_rec_model = load_table_rec_model(table_rec_model_path)
    table_rec_processor = load_table_rec_processor()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"surya模型加载耗时: {elapsed_time:.2f}秒")

    ocr_result = run_surya_ocr(
        IMAGE_PATH, det_model, det_processor, rec_model, rec_processor
    )
    logger.info(f"ocr_result:\n{ocr_result}")

    # text_detection = run_surya_batch_text_detection(
    #     IMAGE_PATH, det_model, det_processor
    # )
    # logger.info(f"text_detection:\n{text_detection}")

    # layout_detection = run_surya_batch_layout_detection(
    #     IMAGE_PATH, layout_model, layout_processor, det_model, det_processor
    # )
    # logger.info(f"layout_detection:\n{layout_detection}")

    # table_detection = run_surya_table_detection(
    #     PDF_PATH,
    #     det_model,
    #     det_processor,
    #     layout_model,
    #     layout_processor,
    #     table_rec_model,
    #     table_rec_processor,
    #     rec_model,
    #     rec_processor,
    # )
    # logger.info(f"table_detection:\n{table_detection}")
