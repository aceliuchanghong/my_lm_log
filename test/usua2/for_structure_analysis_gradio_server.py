import gradio as gr
import os
from dotenv import load_dotenv
import logging
import pandas as pd
from typing import List, Tuple
import random

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def on_select(evt: gr.SelectData, data_list):
    selected_index = evt.index
    # 更新 ans_table 为对应索引的 DataFrame
    df = pd.DataFrame(data_list[selected_index])
    out_excel = generate_excel(data=df)
    return df, out_excel


def generate_excel(data):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 写入 Excel 文件

    excel_file = os.path.join(
        os.getenv("upload_file_save_path"), "structure_analysis", "output.xlsx"
    )
    df.to_excel(excel_file, index=False)

    # 返回文件路径
    return excel_file


def generate_random_data(num_rows: int = 12) -> pd.DataFrame:
    names = ["张三", "李四", "王五"]
    ages = [25, 30, 28]
    cities = ["北京", "上海", "广州"]

    data = {
        "测试姓名": [random.choice(names) for _ in range(num_rows)],
        "测试年龄": [random.choice(ages) for _ in range(num_rows)],
        "测试城市": [random.choice(cities) for _ in range(num_rows)],
    }

    return data


def deal_pics_analysis(
    file_path_list: List[str],
) -> Tuple[List[str], List, pd.DataFrame, str]:
    """
    参数:
        file_path_list (List[str]): 文件路径的字符串列表。

    返回:
        包含以下四个元素:
            - new_file_path_list (List[str]): 新生成的文件路径列表。
            - data (List): 输出的数据的列表。(长度与new_file_path_list相同)
            - df_first (pd.DataFrame): 第一个数据框。
            - out_excel_first (str): 输出的 Excel 文件路径。
    """

    new_file_path_list = file_path_list
    # 生成随机数据
    data = [generate_random_data() for _ in range(len(new_file_path_list))]

    # 给一个初始展示的值,此处不要修改
    df_first = pd.DataFrame(data[0]) if data else pd.DataFrame()
    out_excel_first = generate_excel(data=df_first)

    return new_file_path_list, data, df_first, out_excel_first


def create_app():
    with gr.Blocks(theme=gr.themes.Monochrome(), title="构效分析") as demo:
        with gr.Row():
            gr.Image(value="z_using_files/pics/gouxiao.png", label="TORCH", height=150)
        with gr.Row():
            file_original = gr.File(
                file_types=["image"], label="上传图片", file_count="multiple"
            )
            file_original.GRADIO_CACHE = file_default_path
            with gr.Column():
                clear_button = gr.ClearButton(value="清除历史")
                sure_button = gr.Button(value="提交", variant="primary")
                download_button = gr.DownloadButton(label="下载当前页excel数据")
        with gr.Row():
            gallery = gr.Gallery(label="晶体处理后图片", columns=2, height=520)
            ans_table = gr.DataFrame(label="晶体结果数据")
            all_result = gr.Dropdown(
                label="结果list", visible=False, allow_custom_value=True
            )

        gallery.select(
            on_select, inputs=[all_result], outputs=[ans_table, download_button]
        )

        sure_button.click(
            fn=deal_pics_analysis,
            inputs=[file_original],
            outputs=[gallery, all_result, ans_table, download_button],
        )
        clear_button.add(
            [file_original, gallery, ans_table, all_result, download_button]
        )

        return demo


if __name__ == "__main__":
    # export no_proxy="localhost,127.0.0.1"
    # python test/usua2/for_structure_analysis_gradio_server.py
    # nohup python for_structure_analysis_gradio_server.py > UI.log 2>&1 &
    file_default_path = os.path.join(
        os.getenv("upload_file_save_path"), "structure_analysis"
    )
    os.makedirs(file_default_path, exist_ok=True)
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=11270, share=False)
