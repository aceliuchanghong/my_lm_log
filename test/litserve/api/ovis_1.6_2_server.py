import os
import torch
from modelscope import AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread
from PIL import Image
from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


def preprocess_input(
    conversations, image_input, model, text_tokenizer, visual_tokenizer
):
    # print(conversations, image_input, model, text_tokenizer, visual_tokenizer)
    image_input = Image.open(image_input)
    prompt, input_ids, pixel_values = model.preprocess_inputs(
        conversations, [image_input]
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)

    if image_input is None:
        pixel_values = [None]
    else:
        # print(pixel_values)
        pixel_values = [
            pixel_values.to(
                dtype=visual_tokenizer.dtype, device=visual_tokenizer.device
            )
        ]
    return input_ids, attention_mask, pixel_values


def generate_response(
    input_ids, attention_mask, pixel_values, model, streamer, **gen_kwargs
):
    thread = Thread(
        target=model.generate,
        kwargs={
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "streamer": streamer,
            **gen_kwargs,
        },
    )
    thread.start()
    for new_text in streamer:
        yield new_text


def ovis_chat(
    conversations, image_input, model, text_tokenizer, visual_tokenizer, streamer
):
    input_ids, attention_mask, pixel_values = preprocess_input(
        conversations, image_input, model, text_tokenizer, visual_tokenizer
    )

    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True,
    )

    with torch.no_grad():  # 使用no_grad代替inference_mode，确保代码兼容性
        return generate_response(
            input_ids, attention_mask, pixel_values, model, streamer, **gen_kwargs
        )


def load_model_and_tokenizers(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=1024,
        trust_remote_code=True,
    ).to(device="cuda:1")

    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    streamer = TextIteratorStreamer(
        text_tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    model.generation_config.cache_implementation = None
    return model, text_tokenizer, visual_tokenizer, streamer


# 使用示例
model_name = os.getenv("OVIS_MODEL_PATH")
model, text_tokenizer, visual_tokenizer, streamer = load_model_and_tokenizers(
    model_name
)

# 开始会话
conversations = "<image>\n图片kv提取,json格式"
image_input = "no_git_oic/Snipaste_2024-10-14_13-41-02.png"
for text in ovis_chat(
    conversations, image_input, model, text_tokenizer, visual_tokenizer, streamer
):
    print(text)
