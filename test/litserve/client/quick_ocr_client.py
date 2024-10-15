import requests
import time

start_time = time.time()

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:8109/predict",
    json={
        # pdf的图片拆解之后的url
        "images_path": [
            # "no_git_oic/采购合同4.pdf_show_0.jpg",
            # "./z_using_files/pics/00.png",
            # "./z_using_files/pics/11.jpg",
            # "./z_using_files/pics/00006737.jpg",
            "no_git_oic/page_3.png",
            # "https://www.fmprc.gov.cn/zwbd_673032/jghd_673046/202410/W020241008504386437112.jpg",
            # "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg",
        ],
        "table": "normal",  # 推荐normal 页面复杂使用normal,图像规则使用tsr,tsr是html表格
        # 规则list
        "rule": [
            {
                "entity_name": "10位数条形码号码",
                "entity_format": "2100000010",
                "entity_regex_pattern": "[1-2][0-9]{9}",
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
                "entity_regex_pattern": "S[Oo0][3BA](\d{6}|\d{8})-\d{5}",
            },
        ],
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"耗时: {elapsed_time:.2f}秒")

"""
Status: 200
Response:
[[{"file_name":"00.png","ocr_result":"：月结60天"},{"file_name":"11.jpg","ocr_result":"适合所有肤"},{"file_name":"W020241008522924065946.jpg","ocr_result":"<table><tr><td>75 th 1949-2024 中華人民共和國成立七十五 聯歡晚會 達沃菲建名團 DNCEM</td></tr></table>"}],[{"result":"7813699238","entity_name":"10位数条形码号码"},{"result":"DK","entity_name":"受票单位名称"},{"result":"张祺伟","entity_name":"业务员姓名"},{"result":"160322","entity_name":"开票日期年月"},{"result":"DK","entity_name":"合同-SOB号"}]]
"""
