from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# RapidTable类提供model_path参数，可以自行指定上述2个模型，默认是en_ppstructure_mobile_v2_SLANet.onnx
table_engine = RapidTable(model_path="./test/ocr/ch_ppstructure_mobile_v2_SLANet.onnx")
# ocr_engine = RapidOCR(det_model_path='./test/ocr/ch_PP-OCRv4_det_server_infer.onnx',rec_model_path='./test/ocr/ch_PP-OCRv4_rec_server_infer.onnx')
ocr_engine = RapidOCR()

convert_html_to_md = False

# img_path = "no_git_oic/page_1.png"
# img_path = "no_git_oic/采购合同4.pdf_show_0.jpg"
img_path = "z_using_files/pics/test_pb.png"

ocr_result, _ = ocr_engine(img_path)
table_html_str, table_cell_bboxes, elapse = table_engine(img_path, ocr_result)

# for buck in ocr_result:
#     print(buck[1])
html_table = table_html_str.replace("<html><body>", "").replace("</body></html>", "")
print(html_table)
if convert_html_to_md:
    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
    response = client.chat.completions.create(
        model=os.getenv("HTML_PARSER_MODEL"),
        messages=[{"role": "user", "content": html_table}],
        temperature=0.2,
    )
    print(response.choices[0].message.content)

# python test/ocr/test_rapidocr_table.py
