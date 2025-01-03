import shutil
import gradio as gr
import logging
import os
from dotenv import load_dotenv
from datetime import datetime
import sys
import requests
import time
import json

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
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

from z_utils.check_db import excute_sqlite_sql
from z_utils.get_text_chunk import get_command_run
from z_utils.input_pdf_core import process_file, get_rule_list
from z_utils.sql_sentence import (
    create_rule_table_sql,
    select_rule_sql,
    insert_rule_sql,
    delete_rule_sql,
    select_all_rule_name_sql,
    create_entity_info_sql,
    delete_entity_info_sql,
    insert_entity_info_sql,
    select_rule_file_name_sql,
    select_entity_info_sql,
)


def extract_entities(image_list, rule_name):
    rule_list = get_rule_list(rule_name)
    start_time = time.time()
    response = requests.post(
        f"http://{ip}:8109/predict",
        json={
            "images_path": image_list,
            "table": "normal",
            "rule": rule_list,
        },
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"耗时: {elapsed_time:.2f}秒")
    data = json.loads(response.text)
    ocr_result_list = []
    for ocr_contents in data[0]:
        ocr_result_list.append(ocr_contents["ocr_result"])
    entities = data[1]
    return entities, ocr_result_list


def extract_entity(
    pdf_file_path, image_list, rule, quick_ocr="是", *, progress=gr.Progress()
):
    """
    提取实体的分发,分为长短pdf,图片,是否快速处理
    :param pdf_file_path: pdf路径
    :param image_list: pdf转为图片的list
    :param rule: 提取实体的种类,设置的规则
    :param quick_ocr: 是否快速ocr提取
    :return: entity的list
    """
    if quick_ocr == "是":
        quick_ocr = True
    else:
        quick_ocr = False
    logger.debug(
        f"pdf_file_path:{pdf_file_path},image_list:{image_list},rule:{rule},quick_ocr:{quick_ocr}"
    )
    entities = []
    ocr_result_list = []
    progress(0.1, "图片旋转矫正...")
    if pdf_file_path is None:
        progress(0.9, "提取完成")
        return entities, ocr_result_list
    if pdf_file_path.endswith(".pdf"):
        if len(image_list) < 4:
            progress(0.5, "OCR完成")
            entities, ocr_result_list = extract_entities(image_list, rule)
            progress(0.9, "提取完成")
        else:
            logger.info("长pdf提取开发中")
            ocr_result_list = ["长文本暂未开发"]
            entities = [
                {
                    "sure": False,
                    "rule_name": "提取合同信息规则",
                    "entity_name": "条形码号码",
                    "result": "长pdf提取开发中",
                },
            ]
    else:
        progress(0.5, "OCR完成")
        entities, ocr_result_list = extract_entities(image_list, rule)
        progress(0.9, "提取完成")

    for entity in entities:
        entity["sure"] = False
    logger.debug(f"entities:\n{entities}")
    text_all = "".join(ans for ans in ocr_result_list)
    return entities, gr.update(value=text_all, visible=True)


def create_app():
    with gr.Blocks(title="📋文档实体提取📋", theme=gr.themes.Monochrome()) as demo:
        with gr.Tab(label="📙文档处理"):
            entities = gr.State([])
            logger.debug(f"Entities updated: {entities}")
            with gr.Row():
                gr.Image(
                    label="🤖basic_info",
                    value="z_using_files/pics/ell-wide-light.png",
                    height=250,
                )
            with gr.Row():
                with gr.Column(scale=5):
                    file_original = gr.File(
                        file_count="single",
                        file_types=["image", ".pdf"],
                        label="📕上传文件",
                    )
                    ocr_text = gr.Textbox(label="💡识别结果", visible=True)
                file_original.GRADIO_CACHE = file_default_path
                pic_show = gr.Gallery(
                    label="📙文件预览", scale=5, columns=4, container=True, preview=True
                )
                cut_pic = gr.Dropdown(
                    label="切分图片列表",
                    choices=[],
                    visible=False,
                    allow_custom_value=True,
                )
            gr.Markdown("---")
            with gr.Row():
                with gr.Accordion("🔧基本参数设置", open=False):
                    with gr.Row():
                        rule_option1 = gr.Dropdown(
                            label="1️⃣选择规则",
                            choices=["提取合同信息规则", "提取发票信息规则"],
                            value="提取合同信息规则",
                            interactive=True,
                            info="自定义好规则后需要点击右侧刷新",
                            scale=5,
                        )
                        refresh1 = gr.Button("🧲刷新规则", scale=1)
                    with gr.Row():
                        quick_ocr = gr.Dropdown(
                            label="2️⃣短文档快速识别",
                            choices=["是", "否"],
                            value="是",
                            interactive=True,
                            scale=5,
                            info="选否-则慢速但更精确(此选项仅对页数小于4起效)",
                        )
                        rename_input_file = gr.Dropdown(
                            label="3️⃣是否重命名文件",
                            choices=["是", "否"],
                            value="否",
                            interactive=True,
                            scale=1,
                        )
                key_button = gr.Button(
                    "开始提取",
                    variant="primary",
                    icon="z_using_files/pics/shoot.ico",
                )
            # 结果页面
            gr.Markdown("---")

            @gr.render(inputs=entities)
            def render_entity_result(entity_list):
                if len(entity_list) == 0:
                    return
                user_sure = [entity for entity in entity_list if entity["sure"]]
                wait_user_sure = [
                    entity for entity in entity_list if not entity["sure"]
                ]
                for entity in wait_user_sure:
                    with gr.Row():
                        user_fix_ans = gr.Textbox(
                            label="🖊️" + entity["entity_name"],
                            value=entity["result"],
                            scale=5,
                            interactive=True,
                        )
                        sure_btn = gr.Button("✅确定", scale=1, variant="secondary")

                        def user_make_sure(user_fix_ans, entity=entity):
                            entity["sure"] = True
                            entity["result"] = user_fix_ans
                            logger.debug(f"Entities updated: {entities}")
                            return entity_list

                        sure_btn.click(user_make_sure, user_fix_ans, entities)
                for entity in user_sure:
                    with gr.Row():
                        gr.Textbox(
                            label="🔗" + entity["entity_name"],
                            value=entity["result"],
                            interactive=False,
                        )
                submit_btn = gr.Button("📝提交保存", variant="stop")

                def submit_result(entity_list, file_original, rename_input_file="否"):
                    file_original_name, file_extension = os.path.splitext(file_original)
                    new_file_name = ""
                    if rename_input_file == "是":
                        new_file_name = (
                            "-".join(
                                [
                                    entity["result"]
                                    for entity in sorted(
                                        entity_list, key=lambda x: x["remark"]
                                    )
                                ]
                            )
                            + file_extension
                        )
                        rule_name = entity_list[0]["rule_name"]
                        file_mv2_path = os.path.join(
                            os.getenv("upload_file_save_path", "./upload_files"),
                            "download",
                            rule_name,
                        )
                        os.makedirs(file_mv2_path, exist_ok=True)
                        # 文件到新的路径的逻辑,方便下载
                        shutil.copy(
                            file_original, os.path.join(file_mv2_path, new_file_name)
                        )

                    excute_sqlite_sql(
                        delete_entity_info_sql,
                        (entity_list[0]["rule_name"], os.path.basename(file_original)),
                        False,
                    )
                    for entity in entity_list:
                        insert_entity = {
                            "rule_name": entity["rule_name"],
                            "original_file_name": os.path.basename(file_original),
                            "new_file_name": new_file_name,
                            "entity_name": entity["entity_name"],
                            "result": entity["result"],
                            "latest_modified_insert": datetime.now().strftime(
                                "%Y-%m-%d-%H:%M:%S"
                            ),
                            "remark": entity["remark"],
                        }
                        logger.debug(f"{insert_entity}")
                        try:
                            entity_sql_insert = excute_sqlite_sql(
                                insert_entity_info_sql,
                                (
                                    insert_entity["rule_name"],
                                    insert_entity["original_file_name"],
                                    insert_entity["new_file_name"],
                                    insert_entity["entity_name"],
                                    insert_entity["result"],
                                    insert_entity["latest_modified_insert"],
                                    insert_entity["remark"],
                                ),
                                False,
                            )
                            logger.debug(f"entity_sql插入返回结果:{entity_sql_insert}")
                        except Exception as e:
                            logger.error(e)
                    return []

                submit_btn.click(
                    submit_result,
                    [entities, file_original, rename_input_file],
                    entities,
                )

        file_original.change(
            fn=process_file, inputs=file_original, outputs=[pic_show, cut_pic]
        )
        key_button.click(
            fn=extract_entity,
            inputs=[file_original, cut_pic, rule_option1, quick_ocr],
            outputs=[entities, ocr_text],
        )

        with gr.Tab(label="👉规则设定"):
            with gr.Row():
                gr.Image(
                    label="🤖basic_info",
                    value="z_using_files/pics/ell-wide-light.png",
                    height=250,
                )
            with gr.Row():
                rule_basic_name = gr.Textbox(
                    label="⚙️设置/查询规则名称",
                    placeholder="输入规则名称...eg:提取合同信息规则",
                    autofocus=True,
                    scale=3,
                )
                tasks = gr.State([])
                query_rule = gr.Button("🔍查询规则", scale=1)
                add_rule = gr.Button("🎨新增规则细节", scale=1)

                def add_task(tasks, new_task_name):
                    if len(new_task_name) == 0:
                        return tasks
                    return tasks + [
                        {
                            "name": new_task_name + "_id" + str(len(tasks)),
                            "rendered": False,
                            "entity_name": "",
                            "entity_format": "",
                            "entity_regex_pattern": "",
                            "entity_order": "",
                        }
                    ]

                def query_rule_click(rule_basic_name, tasks=tasks):
                    if rule_basic_name == os.getenv("KEY_WORD", "lchtxdy"):
                        return gr.update(visible=True), tasks
                    else:
                        try:
                            tasks = []
                            entity_tuple_list = excute_sqlite_sql(
                                select_rule_sql, (rule_basic_name,), False
                            )
                            logger.debug(f"entity_tuple_list:{entity_tuple_list}")
                            for i, entity in enumerate(entity_tuple_list):
                                task = {
                                    "name": rule_basic_name + "_id" + str(i),
                                    "rendered": True,
                                    "entity_name": entity[0],
                                    "entity_format": entity[1],
                                    "entity_regex_pattern": entity[2],
                                    "entity_order": entity[3],
                                }
                                tasks.append(task)
                                logger.debug(f"entity:{entity}")
                            logger.debug(f"tasks:{tasks}")
                            return gr.update(visible=False), tasks
                        except Exception as e:
                            logger.error(e)

                add_rule.click(add_task, [tasks, rule_basic_name], [tasks])
            gr.Markdown("---")
            with gr.Column():
                confirm = gr.Button(
                    "🎯提交保存规则",
                    variant="primary",
                    icon="z_using_files/pics/shoot.ico",
                )

                def confirm_click(tasks, rule_basic_name):
                    target_tasks = [
                        task
                        for task in tasks
                        if task["rendered"]
                        and task["name"].split("_id")[0] == rule_basic_name
                    ]

                    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                    excute_sqlite_sql(
                        delete_rule_sql,
                        (target_tasks[0]["name"].split("_id")[0],),
                        False,
                    )
                    for target_task in target_tasks:
                        entity = {
                            "rule_name": target_task["name"].split("_id")[0],
                            "entity_name": target_task["entity_name"],
                            "entity_format": target_task["entity_format"],
                            "entity_regex_pattern": target_task["entity_regex_pattern"],
                            "entity_order": target_task["entity_order"],
                            "rule_state": 1,
                            "latest_modified_insert": current_time,
                            "remark": "暂无",
                        }
                        logger.debug(f"{entity}")
                        try:
                            sql_insert = excute_sqlite_sql(
                                insert_rule_sql,
                                (
                                    entity["rule_name"],
                                    entity["entity_name"],
                                    entity["entity_format"],
                                    entity["entity_regex_pattern"],
                                    entity["entity_order"],
                                    entity["rule_state"],
                                    entity["latest_modified_insert"],
                                    entity["remark"],
                                ),
                                False,
                            )
                            logger.debug(f"sql插入返回结果:{sql_insert}")  # 正常返回[]
                            if sql_insert is None:
                                return [], rule_basic_name + ":新增规则失败"
                        except Exception as e:
                            logger.error(e)
                    return [], rule_basic_name + ":已提交"

                confirm.click(
                    confirm_click,
                    inputs=[tasks, rule_basic_name],
                    outputs=[tasks, rule_basic_name],
                )

            @gr.render(inputs=tasks)
            def render_add_rules(task_list):
                if len(task_list) == 0:
                    return
                # 参考自:https://blog.csdn.net/cxyhjl/article/details/139712016
                incomplete = [
                    task for task in task_list if not task["rendered"]
                ]  # 过滤出渲染未完成的任务
                complete = [task for task in task_list if task["rendered"]]

                for task in incomplete:
                    with gr.Row():
                        entity_name = gr.Textbox(
                            label="🔑要提取的值",
                            placeholder="提取什么值?eg:SOB编号",
                            scale=3,
                            interactive=True,
                        )
                        entity_format = gr.Textbox(
                            label="🔑值的样式",
                            placeholder="该值大概什么样式?eg:SOB20..",
                            scale=3,
                            interactive=True,
                        )
                        entity_regex_pattern = gr.Textbox(
                            label="🔑值的正则表达式",
                            scale=3,
                            interactive=True,
                            placeholder="该值的正则表达式?(可选/若填入则准确值上升)eg:S[Oo0]B[0-9]{1,}-[0-9]{1,}",
                        )
                        entity_order = gr.Textbox(
                            label="🔑值的重命名顺序",
                            placeholder="1,2,3,...",
                            scale=3,
                            interactive=True,
                        )
                        temp_sure_btn = gr.Button(
                            "💪确定", scale=1, variant="secondary"
                        )
                        delete_btn = gr.Button("🖍️删除此行", scale=1, variant="stop")

                        def mark_done(
                            entity_name_value,
                            entity_format_value,
                            entity_regex_value,
                            entity_order,
                            task=task,
                        ):  # 捕获输入值
                            task["rendered"] = True
                            task["entity_name"] = entity_name_value
                            task["entity_format"] = entity_format_value
                            task["entity_regex_pattern"] = entity_regex_value
                            task["entity_order"] = entity_order
                            logger.debug(
                                f"{task['name']},{task['rendered']},{task['entity_name']},{task['entity_format']},{task['entity_regex_pattern']},{task['entity_order']}"
                            )
                            return task_list

                        def delete(task=task):
                            task_list.remove(task)
                            return task_list

                        temp_sure_btn.click(
                            mark_done,
                            [
                                entity_name,
                                entity_format,
                                entity_regex_pattern,
                                entity_order,
                            ],
                            [tasks],
                        )
                        delete_btn.click(delete, None, [tasks])
                for task in complete:
                    with gr.Row():
                        gr.Textbox(
                            label="🔒要提取的值",
                            value=task["entity_name"],
                            interactive=False,
                            scale=3,
                        )
                        gr.Textbox(
                            label="🔒样式",
                            value=task["entity_format"],
                            interactive=False,
                            scale=3,
                        )
                        gr.Textbox(
                            label="🔒正则表达式",
                            value=task["entity_regex_pattern"],
                            interactive=False,
                            scale=3,
                        )
                        gr.Textbox(
                            label="🔒重命名顺序",
                            value=task["entity_order"],
                            scale=3,
                            interactive=True,
                        )
                        delete_btn2 = gr.Button("🖍️删除此行", scale=1, variant="stop")

                        def delete2(task=task):
                            task_list.remove(task)
                            return task_list

                        delete_btn2.click(delete2, None, [tasks])

        with gr.Tab(label="🛸秘密后台", visible=False) as secret_tab:
            gr.Markdown("---")
            last_result = gr.State([])
            with gr.Row():
                rule_option2 = gr.Dropdown(
                    label="🎨选择规则",
                    choices=["提取合同信息规则", "提取发票信息规则"],
                    interactive=True,
                    value="提取合同信息规则",
                    info="自定义好规则后需要点击右侧刷新",
                    scale=5,
                )
                refresh2 = gr.Button("🧲刷新规则", scale=1)
                button_del = gr.Button("🔑删除此规则", scale=1, variant="stop")
            notice = gr.Textbox(visible=False)
            gr.Markdown("---")
            with gr.Row():
                input_command = gr.Textbox(
                    label="🌐输入命令",
                    placeholder="ls",
                    value="ls",
                    interactive=True,
                    scale=5,
                )
                button_command = gr.Button("🔑执行", scale=1, variant="secondary")
            output_command = gr.Textbox(label="✨执行结果", lines=5)
            gr.Markdown("---")
            with gr.Row():
                rule_option3 = gr.Dropdown(
                    label="🧱选择规则", interactive=True, scale=3
                )
                rule_file_name3 = gr.Dropdown(label="🏗️选择文件名", scale=3)
                refresh3 = gr.Button("🚦刷新和文件名", scale=1)
                query3 = gr.Button("🧠查询该文件细节", scale=1)
                button_del3 = gr.Button("💭删除此结果", scale=1, variant="stop")
            notice3 = gr.Textbox(visible=False)

            def get_all_rule_name():
                rule_name_list = []
                all_rule_name = excute_sqlite_sql(
                    select_all_rule_name_sql, should_print=False
                )
                if len(all_rule_name) > 0:
                    for rule_name in all_rule_name:
                        rule_name_list.append(rule_name[0])
                else:
                    rule_name_list = ["提取合同信息规则", "提取发票信息规则"]
                logger.debug(f"rule_name_list:{rule_name_list}")
                return gr.update(value=rule_name_list[0], choices=rule_name_list)

            def delete_rule(rule_name):
                excute_sqlite_sql(delete_rule_sql, (rule_name,), False)
                return gr.Textbox(visible=True, value="已删除:" + rule_name)

            def get_rule_filename(rule_name):
                file_name_list = []
                rule_file_name = excute_sqlite_sql(
                    select_rule_file_name_sql, (rule_name,), should_print=False
                )
                for file_name in rule_file_name:
                    file_name_list.append(file_name[0])
                logger.debug(f"rule_file_name:{rule_file_name}")
                logger.debug(f"file_name_list:{file_name_list}")
                if len(file_name_list) == 0:
                    return gr.update(value=["该规则还未提取过任何实体"])
                return gr.update(choices=file_name_list)

            def delete_rule_filename(rule_name, file_name):
                excute_sqlite_sql(delete_entity_info_sql, (rule_name, file_name), False)
                return gr.Textbox(
                    visible=True, value="已删除:" + rule_name + "," + file_name
                )

            def select_rule_filename_info(rule_name, file_name):
                rule_filename_info = excute_sqlite_sql(
                    select_entity_info_sql, (rule_name, file_name), False
                )
                logger.debug(f"rule_filename_info:{rule_filename_info}")
                new_last_result = []
                for i, entity in enumerate(rule_filename_info):
                    task = {
                        "entity_name": entity[3],
                        "result": entity[4],
                    }
                    new_last_result.append(task)
                logger.debug(f"new_last_result:{new_last_result}")
                return new_last_result

            @gr.render(inputs=last_result)
            def render_entity_result2(last_result_list):
                if len(last_result_list) == 0:
                    return
                user_sure = [entity for entity in last_result_list]
                for entity in user_sure:
                    with gr.Row():
                        gr.Textbox(
                            label="🔗" + entity["entity_name"],
                            value=entity["result"],
                            interactive=False,
                        )

        rule_option3.change(get_rule_filename, rule_option3, rule_file_name3)
        button_del3.click(
            delete_rule_filename, [rule_option3, rule_file_name3], notice3
        )
        query3.click(
            select_rule_filename_info, [rule_option3, rule_file_name3], last_result
        )

        input_command.submit(get_command_run, input_command, output_command)
        button_command.click(get_command_run, input_command, output_command)

        button_del.click(delete_rule, rule_option2, notice)
        refresh3.click(get_all_rule_name, [], rule_option3)
        refresh2.click(get_all_rule_name, [], rule_option2)
        refresh1.click(get_all_rule_name, [], rule_option1)

        query_rule.click(
            query_rule_click, inputs=rule_basic_name, outputs=[secret_tab, tasks]
        )
        rule_basic_name.submit(
            query_rule_click, inputs=rule_basic_name, outputs=[secret_tab, tasks]
        )
    return demo


if __name__ == "__main__":
    """
    export no_proxy="localhost,10.6.6.113,127.0.0.1"
    python test/usua/entity_extract_ui_server.py
    nohup python test/usua/entity_extract_ui_server.py> no_git_oic/entity_extract_ui_server.log &
    """
    ip = "127.0.0.1"
    file_default_path = os.path.join(
        os.getenv("upload_file_save_path", "./upload_files"), "entity_extract"
    )
    os.makedirs(file_default_path, exist_ok=True)
    excute_sqlite_sql(create_rule_table_sql)
    excute_sqlite_sql(create_entity_info_sql)
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("ENTITY_EXTRACT_PORT", 920)),
        share=False,
    )
