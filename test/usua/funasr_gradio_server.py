import gradio as gr
import os
import requests
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


def get_example():
    case = [
        [
            "./z_using_files/mp3/登陆系统,进入层数和K值预测界面.wav",
            "是",
            "是",
            "K值",
        ],
        [
            "./z_using_files/mp3/我是一座孤岛,处在相思之水中.wav",
            "是",
            "否",
            "相思",
        ],
        [
            "./z_using_files/mp3/清晨温柔的光.wav",
            "否",
            "是",
            "清晨",
        ],
    ]
    return case


def run_for_examples(upload_file, timeline_output, spk_output, hot_words):
    return to_asr(upload_file, timeline_output, spk_output, hot_words)


def to_asr(upload_file, timeline_output, spk_output, hot_words):
    mode = "normal"
    need_spk = False
    initial_prompt = hot_words
    ip = "127.0.0.1"
    if timeline_output == "是":
        mode = "timeline"
    if spk_output == "是":
        need_spk = True
    logger.info(
        f"request:upload_file:{upload_file} initial_prompt:{initial_prompt} mode:{mode} need_spk:{need_spk}"
    )

    try:
        response = requests.post(
            f"http://{ip}:{int(os.getenv('FUNASR_PORT'))}/video",
            files={
                "files": open(upload_file, "rb"),
            },
            data={
                "initial_prompt": initial_prompt,
                "mode": mode,
                "need_spk": need_spk,
            },
        )
        if response.status_code == 200:
            content = ""
            for info in response.json()["information"]:
                content += info + "\n"
            file_out = os.path.join(
                result_path, os.path.basename(upload_file).split(".")[0] + ".txt"
            )
            with open(file_out, "w") as f:
                f.write(content)
            return content, gr.update(value=file_out, visible=True)
        else:
            return f"response_status_err_code{response.status_code}", None
    except Exception as e:
        logger.error(
            f"{e}\n ERR_PARAMS:upload_file:{upload_file} initial_prompt:{initial_prompt} mode:{mode} need_spk:{need_spk}"
        )
        return f"error happens:{e}", None


def create_app():
    with gr.Blocks(theme=gr.themes.Ocean(), title="Torch-Asr🎓") as demo:
        with gr.Row():
            gr.Image(
                label="🤖Torch-Asr",
                value="z_using_files/pics/asr_1.png",
                height=280,
            )
        with gr.Row():
            with gr.Column(variant="panel", scale=5):
                file = gr.File(
                    label="🌐上传音频或者视频",
                    file_types=["audio", "video"],
                )
                # file.GRADIO_CACHE = file_default_path
                output_file = gr.File(
                    label="💼结果下载", interactive=False, visible=False
                )
                with gr.Row():
                    timeline_output = gr.Radio(
                        ["是", "否"],
                        label="🔧是否输出时间线",
                        value="是",
                        interactive=True,
                    )
                with gr.Row():
                    spk_output = gr.Radio(
                        ["是", "否"],
                        label="🔧是否输出说话人",
                        value="是",
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
                    convert_button = gr.Button("🚀开始转录", variant="primary")
                    clear_button = gr.ClearButton(value="💬清除历史")
            with gr.Column(variant="panel", scale=5):
                output_content = gr.Textbox(
                    lines=28, show_copy_button=True, label="🔍转录结果"
                )
        with gr.Row():
            gr.Examples(
                label="示例",
                examples=get_example(),
                fn=run_for_examples,
                inputs=[
                    file,
                    timeline_output,
                    spk_output,
                    hot_words,
                ],
                outputs=[output_content, output_file],
            )

        def clear_output():
            return gr.update(visible=False), "是", "是", "会议"

        clear_button.click(
            clear_output, [], [output_file, timeline_output, spk_output, hot_words]
        )
        clear_button.add([file, output_file, output_content])
        convert_button.click(
            fn=to_asr,
            inputs=[
                file,
                timeline_output,
                spk_output,
                hot_words,
            ],
            outputs=[output_content, output_file],
        )
    return demo


if __name__ == "__main__":
    # python test/usua/funasr_gradio_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # nohup python test/usua/funasr_gradio_server.py > no_git_oic/funasr_gradio_server.log &
    file_default_path = os.path.join(os.getenv("upload_file_save_path"), "funasr_video")
    os.makedirs(file_default_path, exist_ok=True)
    result_path = "/mnt/data/asr/result"
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("FUNASR_FRONT_END_PORT")),
        share=False,
        allowed_paths=[result_path],
    )
