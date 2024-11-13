import subprocess


def convert_markdown_to_word(md_file, output_file):
    # 使用 pandoc 将 Markdown 文件转换为 Word 文件
    command = [
        "pandoc",  # pandoc 命令
        md_file,  # 输入的 Markdown 文件路径
        "-o",
        output_file,  # 输出的 Word 文件路径
        "--from",
        "markdown",  # 输入格式为 Markdown
        "--to",
        "docx",  # 输出格式为 Word 文档
    ]

    try:
        # 执行命令
        subprocess.run(command, check=True)
        print(f"转换成功-Word 文件保存在: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")


if __name__ == "__main__":
    # python test/usua/test_pandoc.py
    # md_file = "no_git_oic/60a7c6be-b796-4ac6-bab8-e867abfa2865/auto/60a7c6be-b796-4ac6-bab8-e867abfa2865_tsr.md"
    md_file = "no_git_oic/1db87098-3f2a-4516-8307-e0517a7ec98e/auto/1db87098-3f2a-4516-8307-e0517a7ec98e_tsr.md"
    output_file = "no_git_oic/output2.docx"  # 输出的 Word 文件
    convert_markdown_to_word(md_file, output_file)
