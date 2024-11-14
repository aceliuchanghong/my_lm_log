import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import sys
import time
import json
import concurrent.futures

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
from test.rag.kg_stuff.com_rule_01_fsr import read_docx
from z_utils.get_ai_tools import my_tools
from z_utils.get_text_chunk import chunk_by_LCEL
from test.rag.kg_stuff.prompt import *


def parse_to_list(kg_final_result: str):
    # 将单引号替换为双引号以符合 JSON 格式
    formatted_result = kg_final_result.replace("'", '"')

    # 使用 json.loads 将字符串转换为 Python 列表
    try:
        parsed_list = json.loads(formatted_result)
    except json.JSONDecodeError:
        logger.error("ERR:解析失败，请检查输入格式是否正确。")
        return [
            {"head": "解析错误头", "relation": "list生成失败", "tail": "解析错误尾"}
        ]

    return parsed_list


def process_content(
    content,
    struct,
    previous_content,
    ai_tools,
    model,
    aspect,
    page_number,
    temperature=0.2,
):
    logger.info(f"fixing the {page_number}th chunk")
    messages = [
        {"role": "system", "content": structure_json_prompt.format(ASPECT=aspect)},
        {"role": "user", "content": f"主要参考的当前文档的结构树!!:\n{struct}"},
        {
            "role": "user",
            "content": f"可以用作参考的上页文档部分内容:\n{previous_content}",
        },
        {"role": "user", "content": f"当前页面的详细内容:\n{content}"},
    ]

    # 初次 LLM 处理
    response = ai_tools.llm.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    result = response.choices[0].message.content
    logger.debug(f"json:\n{result}")

    # JSON 转换
    conversion_messages = [
        {"role": "system", "content": structure_kg_prompt},
        {"role": "user", "content": f"主要参考的当前文档的结构树!!:\n{struct}"},
        {"role": "user", "content": f"待转化的json内容:\n{result}"},
    ]

    response = ai_tools.llm.chat.completions.create(
        model=model, messages=conversion_messages, temperature=temperature
    )
    final_result = response.choices[0].message.content
    final_result_list = parse_to_list(final_result)

    # 添加必要的内容
    for res in final_result_list:
        res["content"] = content
        res["id"] = page_number
    logger.debug(f"final_result_list:\n{final_result_list}")

    return final_result_list


def process_docx_files_kg(
    docx_file, ai_tools, model, chunk_size, chunk_overlap, fsr, aspect
):
    struct = fsr
    start_time = time.time()
    content_read = chunk_by_LCEL(
        read_docx(docx_file), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    previous_content = "第一页,暂无内容"
    kg_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for page_number, content in enumerate(content_read, start=1):
            futures.append(
                executor.submit(
                    process_content,
                    content,
                    struct,
                    previous_content,
                    ai_tools,
                    model,
                    aspect,
                    page_number,
                )
            )
            previous_content = content  # 更新上一部分内容

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="kg生成中...",
        ):
            try:
                kg_list.extend(future.result())
            except Exception as e:
                logger.error(f"Error processing content: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"kg生成耗时: {elapsed_time:.2f}秒")
    return kg_list


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/com_rule_02_kg.py
    docx_path = "no_git_oic/com_rule_start/TE-MF-B011培训管理制度V6.1-20230314.docx"
    ai_tools = my_tools()
    chunk_size = int(os.getenv("chunk_size"))
    chunk_overlap = int(os.getenv("chunk_overlap"))
    model = os.getenv("MODEL")
    aspect = "公司培训管理制度"
    fsr = """
公司培训管理制度
├── 公司管理文件
├── 培训管理制度
│   ├── 目的
│   ├── 范围
│   ├── 职责
├── TE-MF-B011
├── 人力资源中心职责
│   ├── 培训计划支持与督导
│   ├── 新员工培训组织
│   ├── 人员资格和能力管理
│   ├── 外部培训机构管理
│   ├── 内训讲师管理
│   ├── 培训效果评估及改进
│   ├── 年度培训总结报告
│   ├── 培训记录与档案管理
│   ├── 企业文化培训内容检查
│   ├── 培训资源整合与分享
├── 各中心/部门负责人职责
│   ├── 年度培训计划制订
│   ├── 培训预算控制及审批
│   ├── 配合人力资源中心工作
│   ├── 岗位培训要求填报改进
│   ├── 业务培训要求提出
│   ├── 培训教材设计与实施
├── 员工职责
│   ├── 积极参与培训课程
│   ├── 提交培训心得报告
│   ├── 培训权利和义务
│   ├── 培训期间纪律遵守
│   ├── 培训资料提交及分享
│   ├── 培训成果应用与转训
│   ├── 个人原因导致培训中止的责任承担
├── 子公司人力资源部职责
│   ├── 制定子公司培训制度
│   ├── 年度培训计划及预算制订
│   ├── 备案股份公司人力资源中心
│   ├── 企业文化或公司简介培训课件制定
│   ├── 培训资源共享
├── 培训经费管理
│   ├── 日常培训经费
│   │   ├── 年度培训总预算
│   │   ├── 培训预算分解
│   │   ├── 培训经费使用范围
│   │   ├── 日常培训经费使用规定
│   ├── 特殊培训经费管理
│   │   ├── 经费清零规则
│   │   ├── 报销规定
│   │   ├── 调休处理
│   │   ├── 内部分享义务
│   │   ├── 公司级培训项目经费管理
│   │   ├── 领导审批权限
├── 学历教育奖励
│   ├── 奖励标准
│   ├── 申请程序
│   ├── 报销规定
├── 培训的策划
│   ├── 能力要求识别
│   ├── 岗位培训要求制定与修订
├── 新生产线岗位培训
│   ├── 培训要求及能力策划
├── 培训类型
│   ├── 内部培训
│   │   ├── 新员工培训
│   │   ├── 在职人员培训
│   ├── 外部培训
│   │   ├── 外部集中培训
│   │   ├── 学历教育
│   ├── 自我发展与提高
├── 培训对象
│   ├── 人员组织
│   ├── 培训引导
│   ├── 培训工具的应用和更新
│   ├── 培训权利和义务
├── 培训档案管理
│   ├── 档案记录
│   │   ├── 教育训练记录
│   │   ├── 个人培训档案
│   │   ├── 年度培训计划表
│   │   ├── 岗位能力评价记录表
│   │   ├── 部门培训需求表
│   │   ├── 实际操作考核记录表
│   │   ├── 季度培训合格率统计分析表
│   │   ├── 季度培训计划跟踪情况表
│   │   ├── 培训心得
│   │   ├── 培训效果评估表
│   │   ├── 培训经费申请表
│   │   ├── 培训协议
│   │   ├── 学历教育奖励申请表
│   │   ├── 新材料、新技术、新工艺培训跟踪表
├── 文件发放范围
│   ├── 公司各部门
│   ├── 各子公司人力资源部
├── TE-QR-G299
"""
    kg_list = process_docx_files_kg(
        docx_path, ai_tools, model, chunk_size, chunk_overlap, fsr, aspect
    )
    logger.info(f"{kg_list}")
