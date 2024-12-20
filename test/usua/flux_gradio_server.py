import gradio as gr
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
from openai import OpenAI

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)


from test.litserve.client.flux_client import generate_image

RESOLUTIONS = [
    (1024, 256),
    (1024, 128),
    (512, 1024),
    (1024, 512),
    (768, 512),
    (768, 1024),
    (1024, 576),
    (576, 1024),
    (1024, 1024),
]


css = """
    #col-container {
        margin: 0 auto;
        max-width: 520px;
    }
    """

default_prompt = "a shepherd boy pointing towards a distant village, with blooming apricot trees, in a serene countryside setting."
sd_system_prompt = """{
    "Role": "你是一位Stable Diffusion提示词专家",
    "Skills": [
        "能够准确提取用户描述中的实体",
        "为每个实体增加详细的描述",
        "根据原文意思将描述联系在一起",
        "将描述转化为地道的英文提示词",
    ],
    "Goal": "接收用户提供的中文诗句或描述，提取其中的实体，增加详细描述，并将其转化为地道的英文提示词。",
    "Instruct": [
        "1. 接收用户提供中文诗句或者中文描述",
        "2. 提取用户给出的描述的实体",
        "3. 对于每一个实体增加细节描述（例如：青花瓷碗-->青彩色的碗,碗上绘制有蓝色倾斜的树木...）",
        "4. 将其描述根据原文意思联系在一起",
        "5. 转化为英文提示词",
    ],
    "Output-Format": "给出地道的英文提示词",
    "Input-Example": "窗含西岭千秋雪",
    "Output-Example": "a painting from a window overlooking distant mountain ranges, with peaks covered in white snow.",
}
"""


def get_example():
    case = [
        "a tiny astronaut hatching from an egg on the moon",
        "a cat holding a sign that says hello world",
        "an anime illustration of a wiener schnitzel",  # 一张维也纳炸牛排的动漫插图
        "a red hair anime girl is twirling her hairr",  # 一个动漫女孩在拨弄她的头发
    ]
    return case


def get_example2():
    case = [
        ["东临碣石,以观沧海"],
        ["清风徐来，水波不兴。举酒属客，诵明月之诗，歌窈窕之章"],
        ["但见悲鸟号古木，雄飞雌从绕林间"],
        ["床前明月光，疑是地上霜。举头望明月，低头思故乡"],
        ["晴川历历汉阳树，芳草萋萋鹦鹉洲"],
        ["乱石穿空，惊涛拍岸，卷起千堆雪"],
        ["少小离家老大回，乡音无改鬓毛衰。儿童相见不相识，笑问客从何处来"],
        ["簌簌衣巾落枣花，村南村北响缲车"],
        ["花褪残红青杏小，燕子飞时，绿水人家绕"],
        ["一个现代感的logo,有数学元素,主要是张扬运动会上面运动健儿的风采"],
    ]
    return case


def infer(prompt, width=1024, height=1024, steps=4):
    if prompt == default_prompt or len(prompt) == 0:
        return "z_using_files/img/flux/牧童遥指杏花村.jpg"
    logger.info(colored(f"prompt:{prompt}", "green"))
    file_path = generate_image(prompt, width, height, steps)
    return file_path


def trans(chinese_prompt):
    if chinese_prompt == "牧童遥指杏花村":
        return default_prompt
    client = OpenAI(
        api_key=os.getenv("API_KEY"), base_url=os.getenv("OLLAMA_CHAT_BASE_URL")
    )
    messages = [
        {"role": "system", "content": sd_system_prompt},
        {"role": "user", "content": chinese_prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=os.getenv("SMALL_MODEL"),
            messages=messages,
            temperature=0.5,
        )
    except Exception as e:
        return default_prompt
    return response.choices[0].message.content


def send_meg_func(english_prompt):
    return english_prompt


def create_app():
    with gr.Blocks(css=css, theme=gr.themes.Ocean(), title="Flux") as demo:
        split_comma = "*"
        gr.Markdown(f"""# FLUX.1-dev 图像生成""")
        with gr.Row():
            with gr.Column():
                result = gr.Image(label="Flux-pic-generation")
                prompt = gr.Text(
                    label="Prompt-提示词",
                    show_label=False,
                    max_lines=100,
                    min_width=320,
                    container=False,
                    placeholder="大约需要24s,请稍加等待",
                    interactive=True,
                )
                run_button = gr.Button("点击生成图片")
            with gr.Column():
                with gr.Accordion("更多参数", open=False):
                    resolution = gr.Dropdown(
                        label="选择分辨率,宽度x长度",
                        choices=[f"{w}{split_comma}{h}" for w, h in RESOLUTIONS],
                        value=f"1024{split_comma}1024",
                    )
                    steps = gr.Slider(
                        label="采样步数", minimum=4, maximum=8, step=1, value=4
                    )
                gr.Examples(
                    label="示例",
                    examples=get_example(),
                    fn=infer,
                    inputs=[prompt],
                    outputs=[result],
                    cache_examples=True,
                    cache_mode="lazy",
                )
                with gr.Row():
                    prompt_generator = gr.Text(
                        label="输入中文提示词,点击右侧提示词生成",
                        interactive=True,
                        value="牧童遥指杏花村",
                    )
                    gen_prompt_button = gr.Button(
                        "点击生成提示词", scale=0, variant="stop"
                    )
                with gr.Row():
                    prompt_english = gr.Text(
                        label="prompt-english",
                        interactive=False,
                    )
                    send_meg = gr.Button(
                        "点击发送提示词至左侧图片生成框", scale=0, variant="stop"
                    )
                gr.Examples(
                    examples=get_example2(),
                    fn=trans,
                    inputs=[prompt_generator],
                    outputs=[prompt_english],
                )

        # 触发生成图片的事件
        run_button.click(
            fn=lambda prompt, resolution, steps: infer(
                prompt,
                int(resolution.split(split_comma)[0]),
                int(resolution.split(split_comma)[1]),
                steps,
            ),
            inputs=[prompt, resolution, steps],
            outputs=[result],
        )
        gen_prompt_button.click(
            fn=trans, inputs=[prompt_generator], outputs=[prompt_english]
        )
        send_meg.click(fn=send_meg_func, inputs=[prompt_english], outputs=[prompt])

    return demo


if __name__ == "__main__":
    """
    export no_proxy="localhost,36.213.66.106,127.0.0.1,112.48.199.202,112.48.199.7"
    python test/usua/flux_gradio_server.py
    nohup python test/usua/flux_gradio_server.py > no_git_oic/flux_gradio_server.log &
    """
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("FLUX_FRONT_END_PORT")))
