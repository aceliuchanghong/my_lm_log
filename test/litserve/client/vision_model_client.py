import requests
import time

start_time = time.time()

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:8110/predict",
    json={
        "images_path": [
            "./upload_files/images/发票签收单2.pdf_show_0.jpg",
            "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg",
        ],
        "rule": {
            "entity_name": "10位数条形码号码",
            "entity_format": "2100000010",
            "entity_regex_pattern": "[0-9]{10}",
        },
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"耗时: {elapsed_time:.2f}秒")
