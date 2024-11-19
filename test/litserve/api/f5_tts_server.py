import re
import tempfile
import torch
from collections import OrderedDict
from importlib.resources import files
import numpy as np
import soundfile as sf
import torchaudio
import os
from dotenv import load_dotenv
import logging
import sys
import litserve as ls


from transformers import AutoModelForCausalLM, AutoTokenizer
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

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
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)


def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    speed=1,
    show_info=gr.Info,
):
    ref_audio, ref_text = preprocess_ref_audio_text(
        ref_audio_orig, ref_text, show_info=show_info
    )

    ema_model = F5TTS_ema_model

    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


def load_f5tts(
    ckpt_path=os.path.join(os.getenv("F5TTS_MODEL_PATH"), "model_1200000.safetensors"),
):
    F5TTS_model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
    )
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


F5TTS_ema_model = load_f5tts()
custom_ema_model, pre_custom_path = None, ""
chat_model_state = None
chat_tokenizer_state = None


class F5TTSAPI(ls.LitAPI):
    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        self.torch_dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            os.getenv("FLOWRENCE_MODEL_PATH"),
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(
            os.getenv("FLOWRENCE_MODEL_PATH"), trust_remote_code=True
        )
        self.prompt = "<DETAILED_CAPTION>"
        self.device = device

    def decode_request(self, request):
        images = request
        return images

    def predict(self, inputs):
        result = {}
        try:
            for name, image in inputs.items():
                pass
            return result
        except Exception as e:
            logger.error(f"error:{e}")
        finally:
            self.clean_memory(self.device)

    def encode_response(self, response):
        return response


if __name__ == "__main__":
    # python test/litserve/api/f5_tts_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # nohup python test/litserve/api/f5_tts_server.py > no_git_oic/f5_tts_server.log &
    api = F5TTSAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=[1])
    server.run(port=int(os.getenv("F5TTS_PORT")))
