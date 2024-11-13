from docx import Document
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import sys
import time

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)
from z_utils.get_ai_tools import my_tools
from z_utils.get_text_chunk import chunk_by_LCEL
from test.rag.kg_stuff.prompt import *

fsr = """
公司培训管理制度
├── 公司管理文件
│   ├── 培训管理制度
│   │   ├── TE-MF-B011
├── 目的
├── 范围
├── 职责
│   ├── 总经理
│   ├── 人资总监、人资分管副总
│   ├── 人力资源中心
│   ├── 各中心/部门负责人
│   │   ├── 年度培训计划制订
│   │   ├── 培训预算控制及审批
│   │   ├── 配合支持培训工作
│   │   ├── 填报改进岗位培训要求
│   │   ├── 业务培训要求提出
│   │   ├── 培训教材设计实施
│   ├── 员工
│   │   ├── 积极参与培训课程
│   │   ├── 提交培训心得报告
│   │   ├── 培训权利和义务
│   │   │   ├── 培训权利
│   │   │   ├── 培训义务
│   ├── 子公司人力资源部
│   │   ├── 制定子公司培训制度
│   │   ├── 年度培训计划及预算制订
│   │   ├── 企业文化培训课件制定
│   │   ├── 共享培训资源
├── 培训经费管理
│   ├── 日常培训经费
│   │   ├── 年度培训总预算
│   │   ├── 培训预算分解
│   │   ├── 使用范围
│   │   ├── 使用规定
│   │   ├── 公司级培训项目经费管理
│   ├── 大额培训费用审批
│   │   ├── 5000元以上且50000元以内
│   │   ├── 50000元以上
├── 学历教育奖励
│   ├── 奖励标准
│   │   ├── 本科（理工科）
│   │   ├── 研究生
│   │   ├── 博士生
│   ├── 申请程序
│   │   ├── 申请表填写及提交
│   │   ├── 审核与发放
├── 培训管理
│   ├── 培训策划
│   │   ├── 能力要求识别
│   │   ├── 培训效果改进建议
│   │   ├── 培训协议及违约金支付标准
│   │   ├── 服务期限约定
│   ├── 员工培训义务
│   │   ├── 认真学习，达到目标
│   │   ├── 遵守纪律
│   │   ├── 提交培训资料和心得
│   │   ├── 应用培训成果
│   │   ├── 承担个人原因导致的费用
├── 培训协议及违约金支付标准
│   ├── 签订条件
│   │   ├── 公司出资培训
│   │   ├── 服务期限约定
│   │   ├── 违约金支付标准
│   │   ├── 多次培训协议并行
├── 相关文件
│   ├── TE-MF-B032《各部门岗位培训要求》
│   ├── TE-MF-B045《高层次人才管理办法》
│   ├── TE-MF-B065《内部培训师管理办法》
├── 相关表单
│   ├── TE-QR-B001 《教育训练记录》
│   ├── TE-QR-B002 《员工个人培训档案》
│   ├── TE-QR-B010 《年度培训计划表》
│   ├── TE-QR-B011 《岗位能力评价记录表》
│   ├── TE-QR-B017 《部门培训需求表》
│   ├── TE-QR-B018 《实际操作考核记录表》
│   ├── TE-QR-B029 《培训心得》
│   ├── TE-QR-B030 《培训效果评估表》
│   ├── TE-QR-B036 《培训经费申请表》
│   ├── TE-QR-B037 《培训协议》
│   ├── TE-QR-B099 《学历教育奖励申请表》
│   ├── TE-QR-B085 《新材料、新技术、新工艺培训跟踪表》
│   ├── TE-QR-G299
├── 发放范围
│   ├── 文件发放至公司各部门及各子公司人力资源部。
"""


def read_docx(file_path):
    document = Document(file_path)
    content = ""
    for paragraph in document.paragraphs:
        content += paragraph.text + "\n"
    return content


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/com_rule_02_json.py
    docx_path = "no_git_oic/com_rule_start"
    ASPECT = "公司培训管理制度"
    ai_tools = my_tools()

    docx_files = [
        os.path.join(docx_path, file)
        for file in os.listdir(docx_path)
        if file.endswith(".docx")
    ]
    for docs_file in docx_files:
        content_read = chunk_by_LCEL(
            read_docx(docs_file),
            chunk_size=int(os.getenv("chunk_size")),
            chunk_overlap=int(os.getenv("chunk_overlap")),
        )
        struct = fsr
        start_time = time.time()

        previous_content = "第一页,暂无内容"  # 用于存储上一部分内容
        page_number = 1  # 初始化页码
        for contents in tqdm(content_read, desc="文件json生成中..."):
            messages = [
                {
                    "role": "system",
                    "content": structure_json_prompt.format(ASPECT=ASPECT),
                },
                {
                    "role": "user",
                    "content": "".join(["主要用作参考的当前文档的结构树:\n", struct]),
                },
                {
                    "role": "user",
                    "content": "".join(
                        ["可以用作参考的上页文档部分内容:\n", previous_content]
                    ),
                },
                {
                    "role": "user",
                    "content": "".join(["当前页面的详细内容:\n", contents]),
                },
            ]
            response = ai_tools.llm.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                temperature=0.2,
            )
            logger.info(
                f"\npage_number:{page_number}\n{response.choices[0].message.content}"
            )
            result = response.choices[0].message.content
            previous_content = contents  # 更新上一部分内容
            page_number += 1  # 增加页码

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"json生成耗时: {elapsed_time:.2f}秒")
        break
