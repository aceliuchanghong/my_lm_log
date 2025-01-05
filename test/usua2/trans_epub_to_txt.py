import os
from ebooklib import epub
from bs4 import BeautifulSoup
import ebooklib


def epub_to_txt(epub_path, output_dir):
    """
    将EPUB文件转换为TXT文件，并清理XML标签和其他不需要的内容
    :param epub_path: EPUB文件路径
    :param output_dir: 输出目录
    :return: 生成的TXT文件路径
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取EPUB文件
    book = epub.read_epub(epub_path)

    # 获取文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(epub_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")

    # 打开输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        # 遍历所有内容项
        for item in book.get_items():
            # 只处理文档类型的内容
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # 使用 BeautifulSoup 解析 HTML/XML 内容
                soup = BeautifulSoup(item.get_content().decode("utf-8"), "html.parser")
                # 提取纯文本并写入文件
                text = soup.get_text()
                f.write(text)
                f.write("\n\n")  # 添加段落分隔

    return output_path


if __name__ == "__main__":
    # uv run test/usua2/trans_epub_to_txt.py
    epub_path = r"C:\\Users\\lawrence\\Downloads\\Reverend_Insanity.epub"
    output_dir = "z_using_files/txt"
    txt_file = epub_to_txt(epub_path, output_dir)
    print(f"转换完成，文件保存在：{txt_file}")
