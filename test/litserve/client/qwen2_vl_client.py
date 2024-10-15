import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
import time

start_time = time.time()

ip = "127.0.0.1"
response = requests.post(
    f"http://{ip}:{int(os.getenv('QWEN2_VL_PORT'))}/predict",
    json={
        "messages": [
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "image",
                    #     "image": "no_git_oic/page_3.png",
                    #     "max_pixels": 512 * 512,
                    # },
                    # {
                    #     "type": "image",
                    #     "image": "no_git_oic/企业微信截图_17288805261441.png",
                    # },
                    {
                        "type": "image",
                        "image": "no_git_oic/Snipaste_2024-10-14_13-41-02.png",
                        "max_pixels": 512 * 512,
                    },
                    {
                        "type": "text",
                        "text": "提取立项必要性分析-背景,市场前景,意义,以json格式返回",
                        # "text": "还原文档,使用markdown格式输出",
                        # "text": "提取:项目编号 | 项目名称 | 项目负责人 | 申请单位 | 项目属性1（内容维度） | 项目属性2（内容维度） | 项目属性3（内容维度） | 项目属性4（需求维度） | 项目属性5(需求维度) | 归口单位 | 成果形式 | 完成形式 | 预计周期-起始时间 | 预计周期-终止时间,以json格式返回",
                    },
                ],
            }
        ],
    },
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"耗时: {elapsed_time:.2f}秒")
