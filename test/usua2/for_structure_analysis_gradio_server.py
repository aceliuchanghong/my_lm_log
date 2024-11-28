import gradio as gr
import os
from dotenv import load_dotenv
import logging
import pandas as pd
from termcolor import colored

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


def deal_pics_analysis(file_path_list):
    """
    修改此函数3处即可
    """
    data = []
    # 数据获取需要修改成自己的实现
    data1 = {
        "测试姓名": [
            "张三",
            "李四",
            "王五",
            "张三",
            "李四",
            "王五",
            "张三",
            "李四",
            "王五",
            "张三",
            "李四",
            "王五",
        ],
        "测试年龄": [25, 30, 28, 25, 30, 28, 25, 30, 28, 25, 30, 28],
        "测试城市": [
            "北京",
            "上海",
            "广州",
            "北京",
            "上海",
            "广州",
            "北京",
            "上海",
            "广州",
            "北京",
            "上海",
            "广州",
        ],
    }
    data2 = {
        "测试姓名3": [
            "张三",
            "李四",
            "王五",
            "张三",
            "李四",
            "王五",
            "张三",
            "李四",
            "王五",
            "张三",
            "李四",
            "王五",
        ],
        "测试年龄3": [25, 30, 28, 25, 30, 28, 25, 30, 28, 25, 30, 28],
        "测试城市3": [
            "北京",
            "上海",
            "广州",
            "北京",
            "上海",
            "广州",
            "北京",
            "上海",
            "广州",
            "北京",
            "上海",
            "广州",
        ],
    }
    # 全部图片的结果数据添加到data这个list里面,需要自己修改
    data.append(data1)
    data.append(data2)
    # 图片添加红色条纹或者其他什么的,然后变成新的图片的逻辑,需要修改
    file_path_list = file_path_list

    # 给一个初始展示的值,此处不要修改
    df_first = pd.DataFrame(data[0])
    out_excel_first = generate_excel(data=df_first)

    return file_path_list, data, df_first, out_excel_first


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
    file_default_path = os.path.join(
        os.getenv("upload_file_save_path"), "structure_analysis"
    )
    os.makedirs(file_default_path, exist_ok=True)
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=11270, share=False)
