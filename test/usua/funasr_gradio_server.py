import gradio as gr
import os

from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app():
    with gr.Blocks(theme=gr.themes.Ocean(), title="Torch-Asr🎓") as demo:
        with gr.Row():
            gr.Image(
                label="🤖Torch-Asr",
                value="z_using_files/pics/asr.png",
                height=300,
            )
        with gr.Row():
            with gr.Column(variant="panel", scale=5):
                file = gr.File(
                    label="🌐上传音频或者视频",
                    file_types=["audio", "video"],
                )
                file.GRADIO_CACHE = file_default_path
                with gr.Row():
                    timeline_output = gr.Radio(
                        ["是", "否"],
                        label="🔧是否输出时间线",
                        value="html",
                        interactive=True,
                    )
                with gr.Row():
                    spk_output = gr.Radio(
                        ["是", "否"],
                        label="🔧是否输出说话人",
                        value="html",
                        interactive=True,
                    )
                with gr.Row():
                    hot_words = gr.Textbox(
                        label="🔥媒体热词",
                        placeholder="会议",
                        value="会议",
                        interactive=True,
                    )
                with gr.Row():
                    convert_button = gr.Button("🚀开始转录")
                    clear_button = gr.ClearButton(value="💬清除历史")
            with gr.Column(variant="panel", scale=5):
                output_content = gr.Textbox(
                    lines=28, show_copy_button=True, label="🎁转录结果"
                )
    return demo


if __name__ == "__main__":
    # python test/usua/funasr_gradio_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # nohup python test/usua/funasr_gradio_server.py > no_git_oic/funasr_gradio_server.log &
    file_default_path = os.path.join(os.getenv("upload_file_save_path"), "funasr_video")
    os.makedirs(file_default_path, exist_ok=True)
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("FUNASR_FRONT_END_PORT")),
        share=False,
    )
