import os
import re
import gradio as gr
from gradio_pdf import PDF
import uuid
import pymupdf
from dotenv import load_dotenv
import logging
import requests
import json
import zipfile
import base64
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
latex_delimiters = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]


def to_pdf(file_path):
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            # 将pdfbytes 写入到uuid.pdf中
            # 生成唯一的文件名
            unique_filename = f"{uuid.uuid4()}.pdf"

            # 构建完整的文件路径
            tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

            # 将字节数据写入文件
            with open(tmp_file_path, "wb") as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)

            logger.info(colored(f"pic->pdf:tmp_file_path:{tmp_file_path}", "green"))

            return tmp_file_path


def to_pdf_byte(file_path):
    try:
        with pymupdf.open(file_path) as f:
            if f.is_pdf:
                pdf_bytes = f.tobytes()
            else:
                pdf_bytes = f.convert_to_pdf()
            return pdf_bytes
    except Exception as e:
        logger.error(colored(f"file to bytes wrong:{file_path}", "green"))
        return None


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r"\!\[(?:[^\]]*)\]\(([^)]+)\)"

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def compress_directory_to_zip(directory_path, output_zip_path):
    """
    压缩指定目录到一个 ZIP 文件。
    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def to_markdown(upload_file, html_md_table="html"):
    convert_html_to_md = False
    parse_method = "auto"
    debug_able = False
    kwargs = {}
    logger.info(colored(f"upload_file:{upload_file}", "green"))

    if html_md_table != "html":
        convert_html_to_md = True
    try:
        kwargs["parse_method"] = parse_method
        kwargs["convert_html_to_md"] = convert_html_to_md
        kwargs["debug_able"] = debug_able

        response = requests.post(
            f"http://127.0.0.1:8116/predict",
            data={"kwargs": json.dumps(kwargs)},
            files={"file": to_pdf_byte(upload_file)},
        )
        logger.info(colored(f"response:{response}", "green"))
        if response.status_code == 200:
            output = response.json()
            output_dir = output["output_dir"]
            logger.info(colored(f"output_dir:{output_dir}", "green"))

            upload_file_save_path = os.getenv("upload_file_save_path", "./upload_files")
            os.makedirs(upload_file_save_path, exist_ok=True)

            all_file_path = os.path.join(
                upload_file_save_path,
                "md_file",
                output_dir,
                parse_method,
            )
            os.makedirs(all_file_path, exist_ok=True)
            # 获取md内容
            with open(
                os.path.join(
                    all_file_path,
                    output_dir + "_tsr.md",
                ),
                "r",
                encoding="utf-8",
            ) as f:
                md_content = f.read()
                html_content = replace_image_with_base64(md_content, all_file_path)
                # print(f"{html_content}")

            # 压缩文件夹
            file_zip_path = os.path.join(
                os.getenv("upload_file_save_path", "./upload_files"), "pdf_zip_path"
            )
            os.makedirs(file_zip_path, exist_ok=True)
            archive_zip_path = os.path.join(
                file_zip_path, os.path.basename(upload_file) + ".zip"
            )
            zip_archive_success = compress_directory_to_zip(
                all_file_path, archive_zip_path
            )
            if zip_archive_success == 0:
                logger.info("压缩成功")
            else:
                logger.error("压缩失败")
            return html_content, md_content, archive_zip_path
        else:
            raise Exception(response.text)
    except Exception as e:
        logger.error(f"File: {upload_file} - : {e}")
        return None, None, None


def create_app():
    with gr.Blocks(title="pdf-md✨✨") as demo:
        with gr.Row():
            gr.Image(
                label="Torch-pdf-converter",
                value="z_using_files/pics/pdf2md_4.png",
                height=260,
            )
        with gr.Row():
            with gr.Column(variant="panel", scale=5):
                file = gr.File(
                    label="上传pdf或者图片",
                    file_types=[".pdf", ".png", ".jpeg", "jpg"],
                )
                with gr.Row():
                    html_md_table = gr.Radio(
                        ["html", "markdown"],
                        label="输出表格格式选择",
                        value="html",
                        interactive=True,
                    )
                with gr.Row():
                    convert_button = gr.Button("开始转化")
                    clear_button = gr.ClearButton(value="清除历史")
                pdf_show = PDF(label="PDF 预览", interactive=True, height=600)
            with gr.Column(variant="panel", scale=5):
                output_file = gr.File(label="结果压缩包下载", interactive=False)
                gr.Markdown("---")
                with gr.Tabs():
                    with gr.Tab("Markdown 渲染"):
                        md = gr.Markdown(
                            label="Markdown rendering",
                            height=650,
                            show_copy_button=True,
                            latex_delimiters=latex_delimiters,
                            line_breaks=True,
                        )
                    with gr.Tab("Markdown 原文"):
                        md_text = gr.TextArea(lines=30, show_copy_button=True)
        file.change(fn=to_pdf, inputs=file, outputs=pdf_show)
        convert_button.click(
            fn=to_markdown,
            inputs=[
                pdf_show,
                html_md_table,
            ],
            outputs=[md, md_text, output_file],
        )
        clear_button.add([file, md, pdf_show, md_text, output_file])
    return demo


if __name__ == "__main__":
    # python test/usua/minerU_gradio_server.py
    # export no_proxy="localhost,127.0.0.1"
    # nohup python test/usua/minerU_gradio_server.py > no_git_oic/minerU_gradio_server.log &
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        root_path="/Pdf2MdTool",
        server_port=int(os.getenv("MINERU_FRONT_END_PORT", 16842)),
        share=False,
        # auth=[("torch", "torch-pdf-markdown"), ("llch", "txdy")],
        # auth_message="输入账户密码登录(torch/torch-pdf-markdown)",
    )
