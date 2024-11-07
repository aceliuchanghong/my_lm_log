import requests
import argparse
import os


def send_audio_to_whisper(audio_path, language="zh", temperature=0.1, timeline=True):
    url = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3-turbo"
    time_show = ""
    headers = {"Authorization": "bearer 3iQIT7UW994mUKVLLmJDLEI4aJiDnXsy"}
    files = {"audio": open(audio_path, "rb")}
    data = {
        "language": language,
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code == 200:
        result = ""
        result_temp = response.json()
        for segment in result_temp["segments"]:
            start = round(segment["start"], 2)
            end = round(segment["end"], 2)
            if timeline:
                time_show = f"[{start}s -> {end}s] "
            result += time_show + segment["text"] + "\n"
        return result  # 返回响应的JSON数据
    else:
        response.raise_for_status()  # 如果请求失败，抛出异常


if __name__ == "__main__":
    # python test/litserve/api/whisper_large_v3_turbo_server.py --file "D:\机器学习记录\make_video\AI-数学-导论-01\Thursday at 9-30 PM.m4a..wav"
    parser = argparse.ArgumentParser(
        description="Process audio file with optional timeline."
    )
    parser.add_argument(
        "--file", type=str, help="Path to the audio file"
    )  # file为必传参数
    parser.add_argument(
        "--timeline", action="store_true", help="Enable timeline (default: False)"
    )

    args = parser.parse_args()
    result = send_audio_to_whisper(args.file, timeline=args.timeline)
    # 保存
    file_path = os.path.abspath(args.file)
    file_dir = os.path.dirname(file_path)  # 获取文件所在目录
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(file_dir, f"{file_name_without_extension}.txt")
    with open(output_file, "w") as f:
        f.write(result)
    print(f"{result}")
