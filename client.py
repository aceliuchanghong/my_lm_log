import requests
import time

start_time = time.time()

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:8109/predict",
    json={
        "images_path": [
            "./z_using_files/pics/00.png",
            "./z_using_files/pics/11.jpg",
            "./z_using_files/pics/00006737.jpg",
            "https://www.fmprc.gov.cn/zwbd_673032/jghd_673046/202410/W020241008504386437112.jpg",
            "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg",
        ],
        "table": "tsr",
        "rule": [
            {
                "entity_name": "10位数条形码号码",
                "entity_format": "2100000010",
                "entity_regex_pattern": "[0-9]{10}",
            },
            {
                "entity_name": "受票单位名称",
                "entity_format": "XX公司",
                "entity_regex_pattern": "",
            },
            {
                "entity_name": "业务员姓名",
                "entity_format": "",
                "entity_regex_pattern": "",
            },
            {
                "entity_name": "开票日期年月",
                "entity_format": "202009",
                "entity_regex_pattern": "[1-2][0-9]{5}",
            },
            {
                "entity_name": "合同-SOB号",
                "entity_format": "SOB20..-..",
                "entity_regex_pattern": "S[Oo0][BA](\d{6}|\d{8})-\d{5}",
            },
        ],
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"耗时: {elapsed_time:.2f}秒")
