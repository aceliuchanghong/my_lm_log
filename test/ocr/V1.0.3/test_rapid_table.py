import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
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

from rapid_table import RapidTable, VisTable, RapidTableInput
from rapidocr_onnxruntime import RapidOCR
from test.ocr.test_surya import polygon_to_markdown
from test.ocr.test_combine_ocr import create_textline_from_data

if __name__ == "__main__":
    """
    python test/ocr/V1.0.3/test_rapid_table.py
    """
    test_pics = [
        "no_git_oic/企业微信截图_17364777534213.png",
        "no_git_oic/型号1-厚度及铅含量(1).jpg",
        "no_git_oic/采购合同2.pdf_show_0.jpg",
    ]
    test_id = 0
    unitable_model_path = {
        "decoder": "no_git_oic/rapid_table_model/unitable/decoder.pth",
        "encoder": "no_git_oic/rapid_table_model/unitable/encoder.pth",
        "vocab": "no_git_oic/rapid_table_model/unitable/vocab.json",
    }
    start_time = time.time()
    input_config = RapidTableInput(
        model_type="unitable",
        model_path=unitable_model_path,
        use_cuda=True,
        device="cuda:3",
    )
    table_engine = RapidTable(input_config)
    ocr_engine = RapidOCR()
    viser = VisTable()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"模型初始化耗时: {elapsed_time:.2f}秒")

    # 测试表格
    start_time = time.time()
    table_results = table_engine(test_pics[test_id])
    table_html_results = (
        table_results.pred_html.split("table_results:")[0]
        .replace("</body></html>", "")
        .replace("<html><body>", "")
    )
    logger.info(
        colored(
            f"table_html_results:{table_html_results}",
            "green",
        )
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"表格识别耗时: {elapsed_time:.2f}秒")

    # 测试文字
    rapid_ocr_result, _ = ocr_engine(test_pics[test_id])
    text_lines = []
    if rapid_ocr_result is None:
        rapid_ocr_result = []
    for line in rapid_ocr_result:
        text_line = create_textline_from_data(line)
        text_lines.append(text_line)
    markdown_start = polygon_to_markdown(text_lines)
    # text_line.polygon 坐标如:[[1.0, 0.0], [61.0, 0.0], [61.0, 21.0], [1.0, 21.0]]
    # 设为 [a,b,c,d] 则 ==> a坐标是左上角坐标，b坐标是右下角坐标，c坐标是右上角的坐标，d坐标是左上角的坐标
    logger.info(
        colored(
            f"{text_lines[0].polygon},{text_lines[0].confidence},{text_lines[0].text},{text_lines[0].bbox}",
            "green",
        )
    )
    logger.info(colored(f"ocr_result:{markdown_start}", "green"))
