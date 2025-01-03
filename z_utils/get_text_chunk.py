# -*- coding: utf-8 -*-
import os

# pip install -qU langchain-text-splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter
import subprocess
import ast


def run_js_script(js_file, file_path):
    # 使用Node.js执行JavaScript文件
    result = subprocess.run(
        ["node", js_file, file_path], capture_output=True, text=True, encoding="utf-8"
    )

    # 将输出解析为列表
    try:
        output_list = ast.literal_eval(result.stdout)  # 解析为列表
    except (ValueError, SyntaxError):
        output_list = []  # 解析失败时返回空列表

    return output_list


def chunk_by_LCEL(file_path, chunk_size=700, chunk_overlap=300):
    # pip install -qU langchain-text-splitters
    # Load example document
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            state_of_the_union = f.read()
    else:
        # 输入的直接是文本
        state_of_the_union = file_path

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([state_of_the_union])
    new_text_doc = [i.page_content for i in texts]
    return new_text_doc


# def chunk_by_LCEL(file_path, chunk_size=700, chunk_overlap=300):
#     #
#     # Load example document
#     if os.path.exists(file_path):
#         with open(file_path, encoding="utf-8") as f:
#             state_of_the_union = f.read()
#     else:
#         # 输入的直接是文本
#         state_of_the_union = file_path

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     texts = text_splitter.create_documents([state_of_the_union])
#     return texts


def get_command_run(command_str):
    # 拆分命令字符串并执行命令
    command = command_str.split(" ")
    print("run command:", command)
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        result.check_returncode()  # 检查命令是否成功执行
        return result.stdout
    except subprocess.CalledProcessError as e:
        # 捕获命令执行失败的情况并返回错误信息
        return f"Command failed with error: {e.stderr}"
    except Exception as e:
        # 捕获其他异常并返回错误信息
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    md_file_path = r"C:\Users\liuch\Documents\img20240708_16193473_latex.md"

    # output = run_js_script('../z_test/chunk.js', md_file_path)
    output = chunk_by_LCEL(md_file_path)
    for y, i in enumerate(output):
        print(str(y) + ":", i.page_content)

    # print(get_command_run("echo hello world"))
    # print(get_command_run("dir"))
