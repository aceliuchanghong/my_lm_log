# export no_proxy="localhost,127.0.0.1"
import requests

# 图像的URL和文本输入
# image_path = "https://picsum.photos/id/237/536/354"
# text_input = "what shown is that pic?"


image_path = "./upload_files/images/发票签收单2.pdf_show_0.jpg"
text_input = "图片中条形码下面编号是多少?"

# image_path = "z_using_files/pics/00006737.jpg"
# text_input = "图片中条形码是多少?"

# 发送POST请求
response = requests.post(
    "http://127.0.0.1:8927/predict",
    json={"image_path": image_path, "text_input": text_input},
)

# 打印状态码和响应内容
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
