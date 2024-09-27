from docling.document_converter import DocumentConverter
# 已测试,没什么用,md
source = "https://arxiv.org/pdf/2408.09869"  # PDF path or URL
# source = "./z_using_files/paper/大模型综述.pdf"  # PDF path or URL
converter = DocumentConverter()
result = converter.convert_single(source)
print(result.render_as_markdown())  # output: "## Docling Technical Report[...]"
print(result.render_as_doctags())  # output: "<document><title><page_1><loc_20>..."
