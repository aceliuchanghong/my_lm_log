import re


def convert_markdown_images(text):
    # 定义正则表达式模式，匹配Markdown图片链接
    pattern = r"!\[.*?\]\((http.*?)\)"

    # 使用正则表达式替换所有匹配的Markdown图片链接
    converted_text = re.sub(pattern, r"[[img=\1;width=100]]", text)

    return converted_text


# 示例输入
input_text = """
执行标准
通用规范:GJB192A-1998《有可靠性指标的无包封多层片式瓷介电容器总规范》
采购 规范:CASS/27.1-2018《中科院微小卫星创新研究院元器件质保规范一
--卫星用多层瓷介质电容器采购规范》![](http://192.168.180.56:9000/top-knowledge-base/ef25ba8357a26a2a599e7e462821e9efcebfb7bb86be27552cb9ac4b92c69ecc.png)
通用规范:GJB192B-2011《有失效率等级的无包封多层片式瓷介固定电容器通用规范》
详细规范:Q/HJ20056-2014《CCK41型有失效率等级的无包封片式1类多层瓷介固定电容器详细规范》
--卫星用多层瓷介质电容器采购规范》
![xwe厕所](http://192.168.180.56:9000/top-knowledge-base/ef25ba8357a26a2a599e7e462821e9efcebfb7bb86be27552cb9ac4b92c69ec0c.png)
"""

# python z_utils/trans_xunfei_link.py
output_text = convert_markdown_images(input_text)
print(output_text)
