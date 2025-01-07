from vision_parse import VisionParser, PDFPageConfig

# https://mp.weixin.qq.com/s/eRqryT1fI6nynl3vjzJGXA
# https://github.com/aceliuchanghong/vision-parse
# Configure PDF processing settings
page_config = PDFPageConfig(
    dpi=400, color_space="RGB", include_annotations=True, preserve_transparency=False
)

# export no_proxy="localhost,10.6.6.113,10.6.6.199,127.0.0.1"
# Initialize parser with custom page config
ip = "10.6.6.199"
parser = VisionParser(
    model_name="internvl2.5",
    api_key="torch-yzgjhdxfxfyzdjhljsjed5h",
    openai_config={"OPENAI_BASE_URL": f"http://{ip}:8110/v1"},
    temperature=0.7,
    top_p=0.4,
    extraction_complexity=False,
    page_config=page_config,
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"
markdown_pages = parser.convert_pdf(pdf_path)
