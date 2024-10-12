import requests


def send_audio_to_whisper(audio_path, language="zh", temperature=0.2):
    url = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3-turbo"
    headers = {"Authorization": "bearer 3iQIT7xxxI4aJiDnXsy"}
    files = {"audio": open(audio_path, "rb")}
    data = {"language": language, "temperature": temperature}
    response = requests.post(url, headers=headers, files=files, data=data)

    return response.text


# 调用函数示例
result = send_audio_to_whisper("z_using_files/mp3/ylh.wav")
print(result)
