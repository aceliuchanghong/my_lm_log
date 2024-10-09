import os
import litserve as ls
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from dotenv import load_dotenv
import requests


class QuickOcrAPI(ls.LitAPI):
    def setup(self, device):
        self.table_engine = RapidTable(model_path=os.getenv('rapidocr_table_engine_model_path'))
        self.ocr_engine = RapidOCR()
        # self.ocr_engine = RapidOCR(det_model_path='./test/ocr/ch_PP-OCRv4_det_server_infer.onnx',rec_model_path='./test/ocr/ch_PP-OCRv4_rec_server_infer.onnx')

    def decode_request(self, request):
        images_path = request["images_path"]
        tsr = request.get("table", "normal")  # 只能tsr,normal
        if tsr != 'normal':
            tsr = 'tsr'
        local_images_path = []
        for image in images_path:
            if os.path.isfile(image):
                # 如果路径是本地文件
                local_image = image
            else:
                # 如果路径是URL，先下载并保存到本地
                save_dir = os.path.join(os.getenv('upload_file_save_path'), 'images')
                os.makedirs(save_dir, exist_ok=True)
                local_image_path = os.path.join(save_dir, os.path.basename(image))  # 保存文件路径

                # 下载并保存图片
                response = requests.get(image, stream=True)
                if response.status_code == 200:
                    with open(local_image_path, 'wb') as out_file:
                        out_file.write(response.content)
                    local_image = local_image_path
                else:
                    raise ValueError(f"Failed to download image from {image}")
            local_images_path.append(local_image)

        return tsr, local_images_path

    def predict(self, inputs):
        tsr, local_images_path = inputs
        ocr_output_result = []
        for local_image in local_images_path:
            ocr_result, _ = self.ocr_engine(local_image)
            ss = ''
            if tsr == 'normal':
                for buck in ocr_result:
                    ss += buck[1]
            elif tsr == 'tsr':
                table_html_str, table_cell_bboxes, elapse = self.table_engine(local_image, ocr_result)
                ss = table_html_str.replace("<html><body>", "").replace("</body></html>", "")
            else:
                raise ValueError("Unsupported tsr value: {}".format(tsr))
            ocr_output_result.append(ss)
        return {"output": ocr_output_result}

    def encode_response(self, output):
        return {"output": output["output"]}


if __name__ == "__main__":
    # python test/litserve/api/quick_ocr_api_server.py
    # export no_proxy="localhost,127.0.0.1"
    load_dotenv()
    api = QuickOcrAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=1)
    server.run(port=8109)
