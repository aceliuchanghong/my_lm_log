import gradio as gr
import numpy as np
import random
import os
from dotenv import load_dotenv
import logging
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        output_type="pil",
        good_vae=good_vae,
    ):
        yield img, seed


def create_app():
    with gr.Blocks(css=css) as demo:

        with gr.Column(elem_id="col-container"):
            gr.Markdown(f"""## FLUX.1-[dev]""")

            with gr.Row():
                prompt = gr.Text(
                    label="Prompt", max_lines=1, placeholder="提示词", container=False
                )
                run_button = gr.Button("Run", scale=0)
            result = gr.Image(label="Result", show_label=False)

            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(
                    label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=15,
                        step=0.1,
                        value=3.5,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )
            gr.Examples(
                examples=examples,
                fn=infer,
                inputs=[prompt],
                outputs=[result, seed],
                cache_examples="lazy",
            )
        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[
                prompt,
                seed,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[result, seed],
        )
    return demo


if __name__ == "__main__":
    examples = [
        "a tiny astronaut hatching from an egg on the moon",
        "a cat holding a sign that says hello world",
        "an anime illustration of a wiener schnitzel",
    ]

    css = """
    #col-container {
        margin: 0 auto;
        max-width: 520px;
    }
    """

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("FLUX_FRONT_END_PORT")))
