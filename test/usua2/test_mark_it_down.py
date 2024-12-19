from markitdown import MarkItDown
from openai import OpenAI

# export no_proxy="localhost,36.213.66.106,127.0.0.1"
# python test/usua2/test_mark_it_down.py
# 效果一般
ip = "127.0.0.1"
client = OpenAI(
    api_key="torch-yzgjhdxfxfyzdjhljsjed5h", base_url=f"http://{ip}:8110/v1/"
)
md = MarkItDown(llm_client=client, llm_model="gpt-4o")
result = md.convert("no_git_oic/采购合同2.pdf_show_0.jpg")
print(result.text_content)
# result = md.convert("no_git_oic/2015.xlsx")
# print(result.text_content)
