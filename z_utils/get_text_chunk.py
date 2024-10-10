# -*- coding: utf-8 -*-
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


if __name__ == "__main__":
    output = chunk_by_LCEL("md_file_path")
    print(output)
