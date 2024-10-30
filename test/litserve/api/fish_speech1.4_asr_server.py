import httpx
import ormsgpack
import argparse

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
    result = response.json()

    # 打印结果
    print(f"Transcribed text: {result['text']}")
    print(f"Audio duration: {result['duration']} seconds")

    for segment in result["segments"]:
        print(f"Segment: {segment['text']}")
        print(f"Start time: {segment['start']}, End time: {segment['end']}")

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
