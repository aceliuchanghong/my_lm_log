import torch
import filetype
import re
import json, uuid
import litserve as ls
from unittest.mock import patch
from fastapi import HTTPException
from magic_pdf.tools.common import do_parse
from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
from openai import OpenAI

# pip install rapidocr_onnxruntime,rapid-table,rapid-orientation
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
import os
from dotenv import load_dotenv
import logging
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)
from z_utils.rotate2fix_pic import detect_text_orientation

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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


def replace_images(md_content, img_dict):
    def replacer(match):
        img_path = match.group(2)
        img_desc = match.group(1)
        if img_path in img_dict:
            replacement = img_dict[img_path]
            if img_desc:
                replacement = f"*{img_desc}:*\n{replacement}\n"
            return replacement
        return match.group(0)  # 如果图片不在字典中，保持原样

    pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    return pattern.sub(replacer, md_content)


def parse_minerU_middle_json(middle_json_file_path):
    # 读取JSON文件,获取表格图片位置
    with open(middle_json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    table_image_list = []
    for page_info in data["pdf_info"]:
        for page_block in page_info["preproc_blocks"]:
            if page_block["type"] == "table":
                for table_image in page_block["blocks"]:
                    if table_image["type"] == "table_body":
                        name = table_image["lines"][0]["spans"][0]["image_path"]
                        table_image_list.append(name)
    logger.debug(f"所有的表格图片: {table_image_list}")
    return table_image_list


class MinerUAPI(ls.LitAPI):
    def __init__(
        self, output_dir=os.path.join(os.getenv("upload_file_save_path"), "md_file")
    ):
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
            print(f"MinerU Model initialization complete!")
            self.ocr_engine = RapidOCR()
            self.table_engine = RapidTable(
                model_path=os.getenv("rapidocr_table_engine_model_path")
            )
            self.client = OpenAI(
                api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL")
            )

    def decode_request(self, request):
        file = request["file"].file.read()
        kwargs = json.loads(request["kwargs"])
        assert filetype.guess_mime(file) == "application/pdf"
        return file, kwargs

    def predict(self, inputs):
        try:
            pdf_name = str(uuid.uuid4())
            convert_html_to_md_param = inputs[1]["convert_html_to_md"]
            del inputs[1]["convert_html_to_md"]
            do_parse(self.output_dir, pdf_name, inputs[0], [], **inputs[1])

            files_path = os.path.join(
                os.getenv("upload_file_save_path"),
                "md_file",
                pdf_name,
                inputs[1]["parse_method"],
            )
            # 开始获取md里面的表格图片
            middle_json_name = os.path.join(files_path, pdf_name + "_middle.json")
            table_image_list = parse_minerU_middle_json(middle_json_name)
            print(f"所有的table_image_list:{table_image_list}")
            # 图片旋转归正
            rotate_images_path = os.path.join(files_path, "rotate_images")
            os.makedirs(rotate_images_path, exist_ok=True)
            for image in table_image_list:
                detect_text_orientation(
                    os.path.join(files_path, "images", image), rotate_images_path
                )
            image_list = [
                os.path.join(rotate_images_path, file)
                for file in os.listdir(rotate_images_path)
                if file.endswith(".jpg")
            ]
            print(f"旋转完成的image_list:{image_list}")
            # 表格识别
            table_results = {}
            for image in image_list:
                ocr_result, _ = self.ocr_engine(image)
                table_html_str, table_cell_bboxes, elapse = self.table_engine(
                    image, ocr_result
                )
                table_code = table_html_str.replace("<html><body>", "").replace(
                    "</body></html>", ""
                )
                convert_html_to_md = convert_html_to_md_param
                if convert_html_to_md:
                    response = self.client.chat.completions.create(
                        model=os.getenv("HTML_PARSER_MODEL"),
                        messages=[{"role": "user", "content": table_code}],
                        temperature=0.2,
                    )
                    table_code = response.choices[0].message.content
                table_results["images/" + os.path.basename(image)] = table_code
            print(f"表格识别完成table_results:{table_results}")
            # 替换md
            original_md = os.path.join(files_path, pdf_name + ".md")
            new_md_path = os.path.join(files_path, pdf_name + "_tsr.md")
            with open(original_md, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = replace_images(content, table_results)
            with open(new_md_path, "w", encoding="utf-8") as file:
                file.write(new_content)
            print(f"替换md完成:{new_md_path}")

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
    # nohup python test/litserve/api/minerU_server.py > no_git_oic/minerU_server.log &
    server = ls.LitServer(MinerUAPI(), accelerator="gpu", devices=[3])
    server.run(port=int(os.getenv("MINERU_SERVER_PORT")))
