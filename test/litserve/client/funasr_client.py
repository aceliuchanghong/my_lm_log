import requests
import time
import os

start_time = time.time()

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:{int(os.getenv('FUNASR_PORT'))}/video",
    files={
        "files": open("no_git_oic/AI_FONT.mp3", "rb"),
    },
    data={
        "initial_prompt": "会议",
        "mode": "timeline",  # normal,timeline
        "need_spk": True,  # True,False
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"耗时: {elapsed_time:.2f}秒")

"""
python test/litserve/client/funasr_client.py
export no_proxy="localhost,112.48.199.202,127.0.0.1"
"""
