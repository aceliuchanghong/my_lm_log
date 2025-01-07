import requests
import time
import os

start_time = time.time()
headers = {"Authorization": "Bearer torch-yzgjsvedioyzdjhljsjed5h"}

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:{int(os.getenv('OCR_PORT', 8124))}/predict",
    json={
        "images_path": [
            "z_using_files/pics/test_pb.png",
            # "z_using_files/pics/image_1.png",
            # "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg",
        ],
        "mode": "markdown",  # normal,markdown,html
    },
    headers=headers,
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"耗时: {elapsed_time:.2f}秒")

"""
python test/litserve/client/pure_ocr_client.py
export no_proxy="localhost,127.0.0.1"
"""
