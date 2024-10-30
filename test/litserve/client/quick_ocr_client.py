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
            "no_git_oic/采购合同4.pdf_show_0.jpg",
            # "https://www.fmprc.gov.cn/zwbd_673032/jghd_673046/202410/W020241008504386437112.jpg",
            # "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg",
        ],
        "table": "normal",
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
python test/litserve/client/quick_ocr_client.py
export no_proxy="localhost,112.48.199.202,127.0.0.1"
Status: 200
Response:
 [[{"file_name":"采购合同4.pdf_show_0.jpg","ocr_result":"Frcm: TORCH Page:3 0f 42021-02-0000:\nS0B202102-14237\n采购合同\n2100000013 需方：秩臻科技股份有限公司 合同编号：\n供方：福建火炬电子科技股份有限公司\n签订时间：2021-2-4 一、货物名称、货物型号、数量、金额：' ' ' ' '\nFrom:TORCH Page:3 Of 4 2021-02-0000:\nSOB202102-14237\n采购合同\n需方:秩蕴科技股份有限公司 合同编号:\n供方:福建火炬电子科技股份有限公司\n签订时间: 2021-2-4 一、货物名称、货物型号、数量、金额:\n金 额 货物型号 含税車价 货物名 交货 数   暮 单位 税率 项目 项次 (元) |流 转 号 时间 称  में है (元) 封装\n← ¥1, 275.00 电容 13% 850 ¥1. 50 40 I\n← ¥6. 150.00 13% 电容 4100 ¥1. 50 2 40\n← 3. 电容 1 3% ¥2. 50 ¥1, 125.00 450 40\nr 13% 电容 46.00 ¥6, 900.00 4 1150 40\n← 电容 5 ¥4. 80 ¥1, 200.00 1 3% 250 40\n电容 ← ¥7, 150.00 1 3% I100 ¥6. 50 0 40\n← 电容 7 ¥2, 500.00 13% 250 ¥10.00 40\n← 电容 ¥2,000.00 8 200 ¥10.00 13% 40\n← ¥10, 200. 00 电容 13% 9 150 468.00 40\n电容 r 13% 2100 ¥3, 150.00 10 ¥1. 50 40\n详细技术指标见原厂、技术规格/标准 合计(小写):  कै\n(人民币) 合同总计(大写):\n二、双方责任: 供方保证在合同规定期限内交货，并保证所提供的产品是原厂原包装新货:需方按照原厂技术规格标准、加工图\n纸、附件等进行产品验收和在合同规定的期限内支付货款;\n三、交货地点及运输方式:供方自定运输方式并承担运输费,运送至 :运输的在途货物贸损、灭先\n四、付款方式:□分期付款:预付第一期,货到验收符合合同约定后支付第二其\n第一期 _%,合计(小写):\n%,合计(小写): 第二期\n_%,合计(小写)_ 第三期\n■ 货到付款:货到验收符合合同约定后,需方承诺在3个且 成 全 款 支\n□预付款;合同签订后,需方预付人民币(小写)。"}],[{"result":"2100000013","entity_name":"10位数条形码号码"},{"result":"秩臻科技股份有限公司","entity_name":"受票单位名称"},{"result":"DK","entity_name":"业务员姓名"},{"result":"202102","entity_name":"开票日期年月"},{"result":"SOB202102-14237","entity_name":"合同-SOB号"}]]
"""
