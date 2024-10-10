import time
from dotenv import load_dotenv
import os
import logging
from openai import OpenAI
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.get_json import parse_json_markdown

load_dotenv()


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


def get_entity_result(client, user_prompt, Basic_info=""):
    system_prompt = """你是一个OCR文档结果提取信息专家
## 技能
- 擅长从有瑕疵的OCR文档中提取信息
- 能够自动识别修复OCR识别中的错别字
- 善于优化信息提取的准确性
## 行动
根据输入的待提取实体的信息和提供的OCR文档结果，提取和校正重要信息
## 约束
输出需符合以下限制：
1. 无法提取到正确匹配值时，answer应为"DK"
2. 结果以JSON格式回复
3. OCR结果不一定准，可能需要自动修复错别字
## 示例输出
{
    "question": "提取世界上最高的山的名字",
    "answer": "珠穆朗玛峰"
}
"""
    prompt = (
        ("## 基本信息:\n" + Basic_info if len(Basic_info) > 10 else "")
        + "\n## 要求:\n"
        + user_prompt
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    start_time = time.time()
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    ans_temp = response.choices[0].message.content

    logger.info(f"user-prompt:\n{prompt}")
    logger.info(f"LLM回答耗时: {elapsed_time:.2f}秒")
    logger.debug(f"大模型response:{response}")
    return parse_json_markdown(ans_temp)


if __name__ == "__main__":
    content = """
    富强有限公司购销合同316000063410TE-QR·S0B201609-02720本合同共页第页签订地点：签订日期：2016年09月30日合同号：I9序号型号规格数量单价总价供货日期执行标准厂家码备注00.053.502016/10/13CB/T2693-20014t700.088.CO2016/10/15GB/T2693-2001!42CC41-D603-CG-50Y-10F-J(N)1009400.052.0020:6/10/13CB/T2693-200134CC41-0402-CG-50V-10pF-J(N)1000.0454.502016/10/15GB/T2693-20015CI41G-0603-2X1-50Y-1.0μF-&(N1000.110.002016/10/15G8/T2693-2001HUAKUN6CT41C-0402-2X1-50V-0.022μP-K(N)1C00.056.002016/10/15GB/T2693-20017CCA1-0402-CG-50Y-680pF-J(N)250.065E.632016/10/13GB/T2693-20018CC41-0603-OG-50Y-8.2pF-C(N)308002.402016/10/13CB/T2693-2009RC0603-10C0F-TC010.0151.502016/10/15CB/T6729-200310RC0603-3300F-T100S10'0L.502016/10/15CB/T5729-200311RC0603-82C0F-T1000.0151.502016/10/15GB/T5729-2003L12RCOE03-000-T1000.0151.502016/10/15GB/T5729-200313RC0402-1403F-T1000.0151.502016/10/15GB/T5729-2003!14RC0402-1562F-T1000.0151.502016/10/15GB/T5729-200315RC0603-IBR0F-T1000.0151,502016/10/15GB/T5729-2003!16RC0603-3C00F-T1000.0151.502016/10/15GB/T5729-200317RC0402-7322F-T1000.0151.502016/10/15Ca/T5729-200318RC0402-7872F-T1000.0151.502016/10/15GB/T5729-2003合计（大写）：伍拾叁元等叁分合计（小写）：53.030执行标准或质量要求：结算方式：30天付款运输提货方式：
    """
    user_prompt = (
        "提取合同-SOB号,结果案例:SOB20..-..,结果正则:S[Oo0][BA][0-9]{6}-[0-9]{5}"
    )
    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
    ans = get_entity_result(client=client, user_prompt=user_prompt, Basic_info=content)
    print(ans)
