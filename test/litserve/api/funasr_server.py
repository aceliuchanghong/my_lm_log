import torch
import re
import litserve as ls
from funasr import AutoModel
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
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

result_path = "/mnt/data/asr/result"
video_path = "/mnt/data/asr/video"
# 模型路径
model_paths = {
    "asr_model": "/mnt/data/asr/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
    "vad_model": "/mnt/data/asr/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_model": "/mnt/data/asr/punc_ct-transformer_cn-en-common-vocab471067-large",
    "spk_model": "/mnt/data/asr/speech_campplus_sv_zh-cn_16k-common",
}


def process_sentences(res, mode="normal"):
    # 检查输入有效性
    if not res or "sentence_info" not in res[0]:
        print("\n##################")
        print(res)
        print("##################\n")
        return (
            [(0, "没有人说话")]
            if mode == "normal"
            else [{"text": "没有人说话", "start": 0, "end": 9999, "spk": 0}]
        )

    result = []
    current_sentence = None
    current_speaker = None

    if mode == "normal":
        for sentence_info in res[0]["sentence_info"]:
            speaker = sentence_info["spk"]
            text = sentence_info["text"]

            if current_speaker is None or speaker != current_speaker:
                if current_sentence:
                    result.append((current_speaker, current_sentence))
                current_speaker = speaker
                current_sentence = text
            else:
                current_sentence += text

        if current_sentence:
            result.append((current_speaker, current_sentence))

    elif mode == "timeline":
        for sentence_info in res[0]["sentence_info"]:
            speaker = sentence_info["spk"]
            text = sentence_info["text"]
            start = sentence_info["start"]
            end = sentence_info["end"]

            if current_speaker != speaker:
                if current_sentence:
                    result.append(current_sentence)
                current_speaker = speaker
                current_sentence = {
                    "text": text,
                    "start": start,
                    "end": end,
                    "spk": speaker,
                }
            else:
                current_sentence["end"] = end
                current_sentence["text"] += text

        if current_sentence:
            result.append(current_sentence)
    return result


def format_timestamp(timestamp):
    # 将时间戳转换为整数分钟和秒
    minutes, seconds = divmod(timestamp / 1000, 60)
    # 格式化为字符串，保留两位小数
    formatted_time = f"{minutes * 60 + seconds:05.2f}s"
    return formatted_time


class FunASRAPI(ls.LitAPI):
    @staticmethod
    def clean_memory(device):
        import gc

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    def setup(self, device):
        # https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md
        self.ars_model = model_paths["asr_model"]
        self.vad_model = model_paths["vad_model"]
        self.punc_model = model_paths["punc_model"]
        self.spk_model = model_paths["spk_model"]
        self.model = AutoModel(
            model=self.ars_model,
            vad_model=self.vad_model,
            punc_model=self.punc_model,
            spk_model=self.spk_model,
            disable_update=True,
            device=device,
        )

    def decode_request(self, request):
        # print(colored(f"received request:{request}", "green"))
        file_name = request["files"].filename
        file_rec = os.path.join(video_path, file_name)
        with open(file_rec, "wb") as f:
            f.write(request["files"].file.read())

        initial_prompt = request.get("initial_prompt", "会议")
        mode = request.get("mode", "normal")
        need_spk = request.get("need_spk", True)
        if isinstance(need_spk, str):
            need_spk = need_spk.lower() == "true"

        return file_rec, initial_prompt, mode, need_spk

    def predict(self, inputs):
        file_rec, initial_prompt, mode, need_spk = inputs
        print(
            colored(
                f"predict file_rec:{file_rec} initial_prompt:{initial_prompt} mode:{mode} need_spk:{need_spk}",
                "green",
            )
        )
        information = []
        try:
            file_path = file_rec
            # 进行识别操作
            res = self.model.generate(
                input=file_path, batch_size_s=1000, hotword=initial_prompt
            )

            chinese_punctuation_pattern = re.compile(
                r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
            )
            # 调用递归函数开始处理
            processed_sentences = process_sentences(res, mode)
            # 打印结果列表，并删除除了最后一句的中文标点
            if mode == "normal":
                for speaker, sentence in processed_sentences:
                    cleaned_sentence = chinese_punctuation_pattern.sub("", sentence)
                    speak_head = f"spk{speaker}:" if need_spk else ""
                    info = speak_head + cleaned_sentence
                    print(info)
                    information.append(info)
            elif mode == "timeline":
                for _ in processed_sentences:
                    speaker = _["spk"]
                    cleaned_sentence = chinese_punctuation_pattern.sub("", _["text"])
                    start = format_timestamp(_["start"])
                    end = format_timestamp(_["end"])
                    speak_head = f"spk{speaker}:" if need_spk else ""
                    info = (
                        speak_head
                        + "["
                        + start
                        + " -> "
                        + end
                        + "] "
                        + cleaned_sentence
                    )
                    print(info)
                    information.append(info)
            # 将结果保存到文件
            result_file_path = os.path.join(
                result_path, f"{os.path.basename(file_path).split('.')[0]}.txt"
            )
            with open(result_file_path, "w") as f:
                for item in res:
                    # 如果结果是字典对象，则转换为字符串
                    if isinstance(item, dict):
                        item = str(item)
                    f.write(str(item) + "\n")
            return {"message": "Processing complete", "information": information}
        except Exception as e:
            logger.error(
                f"error:{e} \nfile:{file_path}, initial_prompt:{initial_prompt}, mode:{mode}, need_spk:{need_spk}"
            )
        finally:
            self.clean_memory(self.device)

    def encode_response(self, response):
        return response


if __name__ == "__main__":
    # python test/litserve/api/funasr_server.py
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # nohup python test/litserve/api/funasr_server.py > no_git_oic/funasr_server.log &
    server = ls.LitServer(
        FunASRAPI(), api_path="/video", accelerator="cuda", devices=[1, 2]
    )
    server.run(port=int(os.getenv("FUNASR_PORT")))
