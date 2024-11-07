import requests
import argparse
import os


def send_audio_to_whisper(audio_path, out_type="txt", translate=False):
    url = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3-turbo"
    headers = {"Authorization": "bearer 3iQIT7UW994mUKVLLmJDLEI4aJiDnXsy"}
    files = {"audio": open(audio_path, "rb")}
    data = {
        "language": "zh",
        "temperature": 0.1,
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code == 200:
        result = ""
        result_temp = response.json()
        if out_type == "txt":
            for segment in result_temp["segments"]:
                start = round(segment["start"], 2)
                end = round(segment["end"], 2)
                time_show = f"[{start}s -> {end}s] "
                result += time_show + segment["text"] + "\n"
        else:
            for i, segment in enumerate(result_temp["segments"], start=1):
                start = segment["start"]
                end = segment["end"]

                # 将秒转换为hh:mm:ss,ms格式
                start_time = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02},{int((start % 1) * 1000):03}"
                end_time = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02},{int((end % 1) * 1000):03}"

                # 生成SRT格式内容
                result += f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n"
        return result
    else:
        response.raise_for_status()  # 如果请求失败，抛出异常


if __name__ == "__main__":
    # TODO 如果传入视频,自动解析为音频,增加翻译功能(libretranslate)
    # python test/litserve/api/whisper_large_v3_turbo_server.py --file "D:\机器学习记录\make_video\AI-数学-导论-01\Thursday at 9-30 PM.m4a..wav"
    # python test/litserve/api/whisper_large_v3_turbo_server.py --file "D:\机器学习记录\make_video\AI-数学-导论-01\Thursday at 9-30 PM.m4a..wav" --out_type srt
    parser = argparse.ArgumentParser(
        description="Process audio file with optional timeline."
    )
    parser.add_argument(
        "--file", type=str, help="Path to the audio file"
    )  # file为必传参数
    parser.add_argument("--translate", action="store_true", help="translate to eng")
    parser.add_argument("--out_type", default="txt", help="outfile type")

    args = parser.parse_args()
    result = send_audio_to_whisper(
        args.file, out_type=args.out_type, translate=args.translate
    )
    # 保存
    file_path = os.path.abspath(args.file)
    file_dir = os.path.dirname(file_path)  # 获取文件所在目录
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(
        file_dir, f"{file_name_without_extension}.{args.out_type}"
    )
    with open(output_file, "w") as f:
        f.write(result)
    print(f"{result}")
