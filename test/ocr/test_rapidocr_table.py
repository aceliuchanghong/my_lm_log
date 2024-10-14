from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from rapid_table import RapidTable, VisTable

# RapidTable类提供model_path参数，可以自行指定上述2个模型，默认是en_ppstructure_mobile_v2_SLANet.onnx
table_engine = RapidTable(model_path="./test/ocr/ch_ppstructure_mobile_v2_SLANet.onnx")
# ocr_engine = RapidOCR(det_model_path='./test/ocr/ch_PP-OCRv4_det_infer.onnx',rec_model_path='./test/ocr/ch_PP-OCRv4_rec_infer.onnx')
ocr_engine = RapidOCR()
# viser = VisTable()

img_path = "no_git_oic/page_1.png"
# img_path = "no_git_oic/采购合同4.pdf_show_0.jpg"
# img_path = './z_using_files/pics/00.png'

ocr_result, _ = ocr_engine(img_path)
table_html_str, table_cell_bboxes, elapse = table_engine(img_path, ocr_result)

# save_dir = Path("./inference_results/")
# save_dir.mkdir(parents=True, exist_ok=True)

# save_html_path = save_dir / f"{Path(img_path).stem}.html"
# save_drawed_path = save_dir / f"vis_{Path(img_path).name}"
# # hether to visualize the layout results.
# viser(img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path)
for buck in ocr_result:
    print(buck[1])
print(table_html_str.replace("<html><body>", "").replace("</body></html>", ""))
