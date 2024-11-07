import httpx
import ormsgpack
import argparse
import os
from dotenv import load_dotenv
import logging
import re

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# https://fish.audio/go-api/billing/


def transcribe_audio(file_input, language="zh", ignore_timestamps=False):
    """
    Transcribe audio using the Fish Audio API.

    :param file_input: Path to the audio file
    :param language: Language of the audio (default is 'zh')
    :param ignore_timestamps: Whether to ignore precise timestamps (default is False)
    :return: Transcription result including text and duration
    """
    # 读取音频文件
    with open(file_input, "rb") as audio_file:
        audio_data = audio_file.read()

    # 准备请求数据
    request_data = {
        "audio": audio_data,
        "language": language,
        "ignore_timestamps": ignore_timestamps,
    }

    # 发送请求
    with httpx.Client() as client:
        response = client.post(
            "https://api.fish.audio/v1/asr",
            headers={
                "Authorization": "Bearer 233caab7218248c8b97d145eac26e905",
                "Content-Type": "application/msgpack",
            },
            content=ormsgpack.packb(request_data),
        )

    # 解析响应
    if response.status_code != 200:
        result = response.json()
        logger.error(f"ERR:{result}")
        return
    result = response.json()
    # logger.info(f"result:{result}")

    # 打印结果
    logger.info(f"Transcribed text: {result['text']}")
    logger.info(f"Audio duration: {result['duration']} seconds")

    for segment in result["segments"]:
        logger.info(f"Segment: {segment['text']}")
        logger.info(f"Start time: {segment['start']}, End time: {segment['end']}")

    # 保存转录文本文件，文件名与输入文件相同，后缀改为 .txt
    output_file = os.path.splitext(file_input)[0] + ".txt"
    with open(output_file, "w", encoding="utf-8") as text_file:
        result_without_punctuation = re.sub(
            r"([，。])\n", r"\n", re.sub(r"([，。])", r"\n", result["text"])
        )
        text_file.write(result_without_punctuation)

    return result


if __name__ == "__main__":
    # python test/litserve/api/fish_speech1.4_asr_server.py --file_input C:\Users\lawrence\Videos\bandicam\对数\对数.mp3
    parser = argparse.ArgumentParser(description="Process asr generation arguments.")
    parser.add_argument(
        "--file_input",
        default="",
    )
    args = parser.parse_args()
    result = transcribe_audio(args.file_input)
