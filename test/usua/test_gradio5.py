import gradio as gr


def file_upload_change_func(file_input_path):
    text_box1, text_box2, text_box3, text_box4 = "", "", "", ""
    return text_box1, text_box2, text_box3, text_box4


def file_click_func(
    file_input_path,
    text_box5,
    text_box6,
    text_box7,
    text_box8,
    text_box5_1,
    text_box6_1,
    text_box7_1,
    text_box8_1,
    text_box9,
    text_box10,
    text_box11,
    text_box12,
    text_box13,
    text_box14,
    text_box15,
    text_box16,
):

    new_file_path = ""
    return new_file_path


def create_app():
    with gr.Blocks(title="📋文件修改") as demo:
        with gr.Row():
            with gr.Column():
                file_original = gr.File(
                    file_types=[".xls", ".xlsx"],
                    label="上传excel文件",
                )
                choose_box = gr.Dropdown(
                    label="技术要求",
                    choices=choose_box_choices,
                    allow_custom_value=True,
                )
            with gr.Column():
                text_box1 = gr.Textbox(label="报告编号", visible=True)
                text_box2 = gr.Textbox(label="规格型号", visible=True)
                text_box3 = gr.Textbox(label="批号", visible=True)
                text_box4 = gr.Textbox(label="数量", visible=True)
        gr.Markdown("---")
        gr.Markdown("常温初测数据")
        with gr.Row():
            with gr.Group():
                with gr.Column():
                    text_box5 = gr.Textbox(label="最大值1", visible=True)
                    text_box6 = gr.Textbox(label="最小值1", visible=True)
                with gr.Column():
                    text_box7 = gr.Textbox(label="最大值2", visible=True)
                    text_box8 = gr.Textbox(label="最小值2", visible=True)
            with gr.Group():
                with gr.Column():
                    text_box5_1 = gr.Textbox(label="最大值3", visible=True)
                    text_box6_1 = gr.Textbox(label="最小值3", visible=True)
                with gr.Column():
                    text_box7_1 = gr.Textbox(label="最大值4", visible=True)
                    text_box8_1 = gr.Textbox(label="最小值4", visible=True)
            text_box9 = gr.Textbox(label="25°CIR", visible=True)
        gr.Markdown("---")
        gr.Markdown("低温/高温测试数据")
        with gr.Row():
            text_box10 = gr.Textbox(label="数据1", visible=True)
            text_box11 = gr.Textbox(label="数据2", visible=True)
            text_box12 = gr.Textbox(label="数据3", visible=True)
            text_box13 = gr.Textbox(label="数据4", visible=True)
            text_box14 = gr.Textbox(label="125°CIR", visible=True)
        gr.Markdown("---")
        gr.Markdown("常温终测数据")
        with gr.Row():
            text_box15 = gr.Textbox(label="终测数据1", visible=True)
            text_box16 = gr.Textbox(label="终测数据2", visible=True)
        gr.Markdown("---")
        with gr.Row():
            sure_button = gr.Button("🧲确认", scale=1)
            download_button = gr.DownloadButton(label="🎁点击下载", variant="stop")

        file_original.change(
            fn=file_upload_change_func,
            inputs=[file_original],
            outputs=[text_box1, text_box2, text_box3, text_box4],
        )
        sure_button.click(
            fn=file_click_func,
            inputs=[
                file_original,
                text_box5,
                text_box6,
                text_box7,
                text_box8,
                text_box5_1,
                text_box6_1,
                text_box7_1,
                text_box8_1,
                text_box9,
                text_box10,
                text_box11,
                text_box12,
                text_box13,
                text_box14,
                text_box15,
                text_box16,
            ],
            outputs=[download_button],
        )

        return demo


if __name__ == "__main__":
    # export no_proxy="localhost,127.0.0.1"
    choose_box_choices = [
        "要求1",
        "要求1",
        "要求3",
        "要求4",
    ]
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=6011, share=False)
