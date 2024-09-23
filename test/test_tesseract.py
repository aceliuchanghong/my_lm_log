import pytesseract
from PIL import Image

# --psm （页面分割模式）参数来告诉Tesseract文档的结构。
# --oem （OCR引擎模式）参数选择不同的OCR引擎模式，以适应不同的识别任务。
"""
--psm 参数（页面分割模式）：
--psm 6: 假设图像是一个单一的文本块。
--psm 1: 自动检测页面布局。
--psm 4: 假设图片是单列文本。
--oem 参数（OCR 引擎模式）：
--oem 0: 仅使用传统 Tesseract OCR 引擎。
--oem 1: 仅使用 LSTM（神经网络）引擎。
--oem 2: 使用传统和 LSTM 引擎。
--oem 3: 默认模式，自动选择最佳引擎。
"""
custom_config = r'--oem 3 --psm 6'
# image = Image.open('../z_using_files/pics/00006737.jpg')
# image = Image.open('../z_using_files/pics/00111002.jpg')
# image = Image.open('../z_using_files/pics/img.png')
image = Image.open('../z_using_files/pics/00077949.jpg')

# 放大图片，resize() 方法中 (width, height) 是新大小
# 比如放大2倍
# image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
# 识别图片中的文字，并设置简体中文和英文混合识别
text = pytesseract.image_to_string(image, lang='chi_sim+eng', config=custom_config)

# 打印识别出的文字
print(text)
