from vision_parse import VisionParser, PDFPageConfig
import os
from PIL import Image

# https://mp.weixin.qq.com/s/eRqryT1fI6nynl3vjzJGXA
# https://github.com/aceliuchanghong/vision-parse
# Configure PDF processing settings


def convert_to_pdf(file_path):
    image = Image.open(file_path)
    pdf_path = file_path.rsplit(".", 1)[0] + ".pdf"
    image.save(pdf_path, "PDF", resolution=100.0)
    return pdf_path


page_config = PDFPageConfig(
    dpi=400, color_space="RGB", include_annotations=True, preserve_transparency=False
)

# export no_proxy="localhost,127.0.0.1"
# python test/usua2/test_vision-parse.py
ip = "127.0.0.1"
parser = VisionParser(
    model_name="gpt-4o",
    api_key="torch-yzgjhdxfxfyzdjhljsjed5h",
    openai_config={"OPENAI_BASE_URL": f"http://{ip}:8110/v1"},
    temperature=0.7,
    top_p=0.4,
    extraction_complexity=False,
    page_config=page_config,
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"


# Check file extension and convert if necessary
file_extension = os.path.splitext(pdf_path)[1].lower()
if file_extension in [".jpg", ".jpeg", ".png"]:
    pdf_path = convert_to_pdf(pdf_path)

# Continue processing the (possibly converted) PDF
markdown_pages = parser.convert_pdf(pdf_path)
