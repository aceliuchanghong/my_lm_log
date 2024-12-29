from vision_parse import VisionParser, PDFPageConfig

# https://github.com/aceliuchanghong/vision-parse
# Configure PDF processing settings
page_config = PDFPageConfig(
    dpi=400, color_space="RGB", include_annotations=True, preserve_transparency=False
)

# Initialize parser with custom page config
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.7,
    top_p=0.4,
    extraction_complexity=False,
    page_config=page_config,
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"
markdown_pages = parser.convert_pdf(pdf_path)
