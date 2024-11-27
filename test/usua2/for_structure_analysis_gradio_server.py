import gradio as gr
import os
from dotenv import load_dotenv
import logging
import pandas as pd
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def deal_pics_analysis(file_path_list, text_box="500", vram="nm"):
    logger.info(colored(f"{file_path_list}", "green"))
    data = {
        "测试姓名": ["张三", "李四", "王五"],
        "测试年龄": [25, 30, 28],
        "测试城市": ["北京", "上海", "广州"],
    }
    df = pd.DataFrame(data)

    return file_path_list, df


def create_app():
    with gr.Blocks(theme=gr.themes.Monochrome(), title="构效分析") as demo:
        with gr.Row():
            gr.Image(value="z_using_files/pics/gouxiao.png", label="TORCH")
        with gr.Row():
            file_original = gr.File(
                file_types=["image"], label="上传图片", file_count="multiple"
            )
            file_original.GRADIO_CACHE = file_default_path
            with gr.Column():
                text_box = gr.Textbox(
                    label="晶粒长度", placeholder="仅针对单个图片填写"
                )
                vram = gr.Radio(
                    ["nm", "µm"], value="nm", label="晶粒单位", interactive=True
                )
                sure_button = gr.Button(value="提交", variant="huggingface")
        with gr.Row():
            gallery = gr.Gallery(label="晶体处理后图片", columns=2)
            ans_table = gr.DataFrame(label="晶体结果数据")

        sure_button.click(
            fn=deal_pics_analysis,
            inputs=[file_original, text_box, vram],
            outputs=[gallery, ans_table],
        )

        return demo


if __name__ == "__main__":
    # export no_proxy="localhost,127.0.0.1"
    # python test/usua2/for_structure_analysis_gradio_server.py
    file_default_path = os.path.join(
        os.getenv("upload_file_save_path"), "structure_analysis"
    )
    os.makedirs(file_default_path, exist_ok=True)
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=11270, share=False)
