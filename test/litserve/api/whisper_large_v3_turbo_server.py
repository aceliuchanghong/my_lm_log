import requests
import argparse
import os
import ffmpeg
import io
import time
from dotenv import load_dotenv
import logging
from termcolor import colored
import zhconv
from translate import Translator


load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# pip install ffmpeg-python
# pip install zhconv
# pip install translate

# pip install libretranslate
"""
模型文件:https://www.argosopentech.com/argospm/index/
libretranslate --load-only en,zh
模型地址:
C:\\Users\\<用户名>\\.local\\share\\argos-translate\\packages
~/.local/share/argos-translate/packages
"""


def translate_text(word):
    translator = Translator(to_lang="en", from_lang="zh")
    translation = translator.translate(word)
    return translation


def transform2sim(string, tran_type="simple"):
    if tran_type == "simple":
        new_str = zhconv.convert(string, "zh-hans")
    else:
        new_str = zhconv.convert(string, "zh-hant")
    return new_str


def format_time(seconds):
    """将秒数转换为hh:mm:ss,ms格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


# def translate_text(text, source_lang="zh", target_lang="en"):
#     url = "https://libretranslate.com/translate"
#     headers = {"Content-Type": "application/json"}
#     payload = {"q": text, "source": source_lang, "target": target_lang}
#     response = requests.post(url, json=payload, headers=headers)

#     # 确保请求成功
#     if response.status_code == 200:
#         logger.info(f"{response.json()['translatedText']}")
#         return response.json()["translatedText"]  # 返回响应的 JSON 数据
#     else:
#         logger.error(f"Failed.{response.status_code}")
#         return "trans error"


def process_translation(text, translate):
    """处理翻译文本"""
    return f"\n{translate_text(text)}" if translate else ""
    # return f"暂时没翻译接口" if translate else ""


def send_audio_to_whisper(
    audio_path, out_type="txt", translate=False, tran_type="simple", language="zh"
):
    url = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3-turbo"
    headers = {"Authorization": "bearer 3iQIT7UW994mUKVLLmJDLEI4aJiDnXsy"}
    data = {
        "language": language,
        "temperature": 0.1,
    }
    # 获取文件扩展名并转换为小写
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}  # 视频文件扩展名
    audio_extensions = {".wav", ".mp3", ".flac"}  # 音频文件扩展名
    file_extension = os.path.splitext(audio_path)[-1].lower()
    # 处理媒体文件
    start_time = time.time()
    if file_extension in video_extensions:
        try:
            audio_stream, err = (
                ffmpeg.input(audio_path)
                # .output(
                #     "pipe:1", acodec="pcm_s16le", format="wav"
                # )  # 输出到内存,可能会比较大,但是速度很快
                .output("pipe:1", acodec="libmp3lame", format="mp3").run(
                    capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg._run.Error as e:
            logger.error(f"FFmpeg error occurred: {e.stderr.decode('utf-8')}")
            raise
        # 使用 io.BytesIO 将音频流包装成文件对象
        audio_file = io.BytesIO(audio_stream)
        audio_file_size = len(audio_file.getvalue())
    elif file_extension in audio_extensions:
        # 音频文件：直接打开音频文件
        audio_file = open(audio_path, "rb")
        audio_file_size = os.path.getsize(audio_file.name)
    else:
        raise ValueError("文件类型不支持，请提供视频或音频文件。")

    audio_file_size_mb = audio_file_size / (1024 * 1024)
    logger.info(f"获取到的音频大小: {audio_file_size_mb:.2f} MB")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"1.媒体文件获取耗时: {elapsed_time:.2f}秒")

    files = {"audio": audio_file}

    start_time = time.time()
    response = requests.post(url, headers=headers, files=files, data=data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"2.转录耗时: {elapsed_time:.2f}秒")

    # 处理完成后关闭文件流（如果是打开的文件）
    if isinstance(audio_file, io.BufferedReader):
        audio_file.close()

    if response.status_code == 200:
        result = ""
        result_temp = response.json()
        logger.info(f"3.0.翻译开始,共{len(result_temp['segments'])}句")
        for i, segment in enumerate(result_temp["segments"], start=1):
            start = round(segment["start"], 2)
            end = round(segment["end"], 2)
            simple_text = transform2sim(segment["text"], tran_type)

            # 统一处理翻译部分
            start_time = time.time()
            translate_text_out = process_translation(
                transform2sim(segment["text"], "simple"), translate
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"3.{i}.翻译第{i}句耗时: {elapsed_time:.2f}秒")

            if out_type == "txt":
                time_show = f"[{start}s -> {end}s] "
                result += time_show + simple_text + translate_text_out + "\n"
            else:
                start_time = format_time(start)
                end_time = format_time(end)
                # 生成SRT格式内容
                result += f"{i}\n{start_time} --> {end_time}\n{simple_text}{translate_text_out}\n\n"
        logger.info(f"4.开始输出...")
        logger.info(colored(f"\n{result}", "green"))
        return result
    else:
        response.raise_for_status()  # 如果请求失败，抛出异常


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/litserve/api/whisper_large_v3_turbo_server.py --file "D:\机器学习记录\make_video\AI-数学-导论-01\Thursday at 9-30 PM.m4a..wav"
    # python test/litserve/api/whisper_large_v3_turbo_server.py --file "D:\机器学习记录\make_video\AI-数学-导论-01\Thursday at 9-30 PM.m4a..wav" --out_type srt
    """
    python test/litserve/api/whisper_large_v3_turbo_server.py \
        --file no_git_oic/gen_new2.mp3 \
        --out_type txt
        
    python test/litserve/api/whisper_large_v3_turbo_server.py \
        --file no_git_oic/会议纪要操作演示6.mp4 \
        --translate
        
    python test/litserve/api/whisper_large_v3_turbo_server.py \
        --file /mnt/data/llch/fluxgym/flowrence/WG40.mp4 \
        --out_type txt \
        --language en

    python test/litserve/api/whisper_large_v3_turbo_server.py --file "D:\机器学习记录\make_video\AI-数学-对数-02\AI-数学-对数-02.mp4" --translate
    """
    parser = argparse.ArgumentParser(
        description="Process audio file with optional timeline."
    )
    parser.add_argument(
        "--file", type=str, help="Path to the audio file"
    )  # file为必传参数
    parser.add_argument("--translate", action="store_true", help="translate to eng")
    parser.add_argument("--out_type", default="srt", help="outfile type")
    parser.add_argument("--tran_type", default="simple", help="简体繁体")
    parser.add_argument("--language", default="zh", help="语言种类")

    args = parser.parse_args()
    result = send_audio_to_whisper(
        args.file,
        out_type=args.out_type,
        translate=args.translate,
        tran_type=args.tran_type,
        language=args.language,
    )
    # 保存字幕文件
    file_path = os.path.abspath(args.file)
    file_dir = os.path.dirname(file_path)  # 获取文件所在目录
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(
        file_dir, f"{file_name_without_extension}.{args.out_type}"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
