import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
import time
from typing import List

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
from test.ocr.test_combine_ocr import create_textline_from_data, NewTextLine


def polygon_to_markdown(text_lines: List[NewTextLine]) -> str:
    # 计算所有文本行的平均高度
    total_height = 0
    for text_line in text_lines:
        height = text_line.bbox[3] - text_line.bbox[1]
        total_height += height
    average_height = total_height / len(text_lines) if text_lines else 0

    # 动态设置 y_tolerance 为平均高度的一半
    y_tolerance = average_height / 2

    # 首先根据Y坐标和高度将文本行分组，考虑容差值
    lines = []
    for text_line in text_lines:
        y_coord = text_line.bbox[1]
        height = text_line.bbox[3] - text_line.bbox[1]

        # 查找是否存在在容差范围内的Y坐标和高度
        found = False
        for line in lines:
            # 检查Y坐标和高度是否在容差范围内
            if (
                abs(y_coord - line["y"]) <= y_tolerance
                and abs(height - line["height"]) <= y_tolerance
            ):
                line["text_lines"].append(text_line)
                found = True
                break

        # 如果没有找到匹配的行，则创建一个新的行
        if not found:
            lines.append({"y": y_coord, "height": height, "text_lines": [text_line]})

    # 对每一行的文本按照X坐标排序
    sorted_lines = []
    for line in sorted(lines, key=lambda x: x["y"]):
        text_lines = line["text_lines"]
        # 按照bbox的X坐标（即左上角的X坐标）排序
        sorted_line = sorted(text_lines, key=lambda x: x.bbox[0])
        sorted_lines.append(sorted_line)

    # 生成Markdown表格
    markdown_table = []
    headers = []
    alignments = []

    # 生成表头
    for i in range(len(sorted_lines[0])):
        headers.append("Column {}".format(i + 1))
        alignments.append(":---:")  # 默认居中对齐

    markdown_table.append("| " + " | ".join(headers) + " |")
    markdown_table.append("| " + " | ".join(alignments) + " |")

    # 生成表格内容
    for line in sorted_lines:
        row = []
        prev_x_end = -1  # 记录上一个文本的结束X坐标
        for text_line in line:
            # 如果当前文本的起始X坐标与上一个文本的结束X坐标有重叠，则合并到同一单元格
            if text_line.bbox[0] < prev_x_end:
                if row:
                    row[-1] += "<br>" + text_line.text  # 多行文本用 <br> 连接
            else:
                row.append(text_line.text)
            prev_x_end = text_line.bbox[2]  # 更新上一个文本的结束X坐标
        # 将同一行的文本用 " | " 连接起来
        markdown_table.append("| " + " | ".join(row) + " |")

    # 用换行符连接每一行
    markdown_output = "\n".join(markdown_table)

    return markdown_output


if __name__ == "__main__":
    """
    python test/ocr/V1.0.3/test_rapid_table.py
    """
    test_pics = [
        "no_git_oic/企业微信截图_17364777534213.png",
        "no_git_oic/型号1-厚度及铅含量(1).jpg",
        "no_git_oic/采购合同2.pdf_show_0.jpg",
    ]
    test_id = 2
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

    # 测试文字markdown
    start_time = time.time()
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
    # text_line.text 为 OCR 文字
    logger.info(colored(f"ocr_markdown_result:{markdown_start}", "green"))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"markdown耗时: {elapsed_time:.2f}秒")
