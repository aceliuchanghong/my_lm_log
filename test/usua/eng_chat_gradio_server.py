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


def generate_response(audio_in):
    pass


def load():
    return [
        (
            "Here's an audio",
            gr.Audio(
                "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
            ),
        ),
        (
            "Here's an video",
            gr.Video(
                "https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4"
            ),
        ),
    ]


def create_app():
    with gr.Blocks(theme=gr.themes.Ocean(), title="chat🎤-chat🎧") as demo:
        state = gr.State()
        chatbot = gr.Chatbot(
            label="🐧您的口语练习助手", elem_classes="control-height", height=750
        )
        audio_in = gr.Audio(
            label="🚀Speak your question", sources="microphone", type="filepath"
        )
        with gr.Row():
            empty_bin = gr.ClearButton(value="🧹Clear")
            regen_btn = gr.Button("🔄Regenerate")
        button = gr.Button("Load audio and video")
        button.click(load, None, chatbot)

        # audio_in.stop_recording(
        #     generate_response, audio_in, [state, answer, audio_out]
        # ).then(fn=read_response, inputs=state, outputs=[answer, audio_out])
    return demo


if __name__ == "__main__":
    # python test/usua/eng_chat_gradio_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # nohup python test/usua/eng_chat_gradio_server.py > no_git_oic/eng_chat_gradio_server.log &
    file_default_path = os.path.join(os.getenv("upload_file_save_path"), "eng_chat")
    os.makedirs(file_default_path, exist_ok=True)
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("ENG_CHAT_END_PORT")),
        share=False,
    )
