import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import pandas as pd
from openpyxl import Workbook
import shutil

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_employee_folder_and_copy_pic(
    employee_id, employee_name, pic_path, base_dir="no_git_oic/true"
):
    """
    在指定目录下创建员工文件夹，并将图片复制到该文件夹中。

    :param employee_id: 员工ID
    :param employee_name: 员工姓名
    :param pic_path: 图片路径
    :param base_dir: 基础目录，默认为 "no_git_oic/true"
    """
    # 创建文件夹名称
    folder_name = f"{employee_id}_{employee_name}"

    # 拼接完整路径
    full_folder_path = os.path.join(base_dir, folder_name)

    # 如果文件夹不存在，则创建
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)

    # 获取图片文件名
    pic_filename = os.path.basename(pic_path)

    # 拼接目标图片路径
    destination_pic_path = os.path.join(full_folder_path, pic_filename)

    # 复制图片到目标文件夹
    shutil.copy(pic_path, destination_pic_path)

    logger.info(
        colored(
            f"Created folder and copied picture for {employee_name} to {full_folder_path}",
            "blue",
        )
    )


# 获取路径 no_git_oic/员工照片-20221227 下面全部的图片 后缀名小写为.jpg或者.jpeg或者.png,返回list
def get_all_files():
    path = "no_git_oic/员工照片-20221227"
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if (
                file.lower().endswith(".jpg")
                or file.lower().endswith(".jpeg")
                or file.lower().endswith(".png")
            ):
                file_list.append(os.path.join(root, file))
    return file_list


# 读取 no_git_oic/通讯录.xlsx 第一个sheet里面的F列的全部员工姓名，返回list
def get_employee_names_and_ids():
    # 读取 Excel 文件
    file_path = "no_git_oic/通讯录.xlsx"
    df = pd.read_excel(file_path, sheet_name=0)  # sheet_name=0 表示第一个 sheet

    # 获取 E 列（编号）和 F 列（员工姓名）的数据
    employee_ids = df.iloc[:, 4].tolist()  # E 列是第 5 列，索引为 4
    employee_names = df.iloc[:, 5].tolist()  # F 列是第 6 列，索引为 5

    # 将编号和姓名组合成字典
    employee_dict = dict(zip(employee_ids, employee_names))
    # 返回字典
    return employee_dict


def write_to_excel(sheet, row, employee_id, employee_name, employee_pic_addr):
    """
    将 employee_id 和 employee_name 写入 Excel 的指定行
    :param sheet: Excel 工作表对象
    :param row: 写入的行号
    :param employee_id: 员工ID
    :param employee_name: 员工姓名
    """
    sheet[f"A{row}"] = employee_id  # 写入 A 列
    sheet[f"B{row}"] = employee_name  # 写入 B 列
    sheet[f"H{row}"] = employee_pic_addr  # 写入 H 列


if __name__ == "__main__":
    # python z_utils/gen_face_recog.py
    employee_name_pic_list = {}
    employee_pic_list = get_all_files()
    for x in employee_pic_list:
        name = (
            os.path.basename(x).split("-")[0]
            if len(os.path.basename(x).split("-")[0]) <= 5
            else os.path.basename(x).split("-")[1].split(".")[0]
        )
        employee_name_pic_list[name] = x
    logger.info(
        colored(f"{employee_name_pic_list},{len(employee_name_pic_list)}", "green")
    )
    excel_employee_dic = get_employee_names_and_ids()
    logger.info(colored(f"{excel_employee_dic},{len(excel_employee_dic)}", "green"))
    with open("employee_name_pic_list.py", "w", encoding="utf-8") as file:
        file.write(f"employee_name_pic_list = {employee_name_pic_list}")
    with open("excel_employee_dic.py", "w", encoding="utf-8") as file:
        file.write(f"excel_employee_dic = {excel_employee_dic}")

    # 创建 Excel 工作簿和工作表
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Employee Data"

    # 输出 excel_employee_dic 里面的员工姓名+ employee_pic_list 对应的图片路径
    i = 1
    for employee_id, employee_name in excel_employee_dic.items():
        for pic_path in employee_name_pic_list.keys():
            if employee_name == pic_path:
                logger.info(
                    colored(
                        f"{i}: {employee_id}: {employee_name}: {employee_name_pic_list[employee_name]}",
                        "green",
                    )
                )
                write_to_excel(
                    sheet,
                    i,
                    employee_id,
                    employee_name,
                    "true/"
                    + employee_id
                    + "_"
                    + employee_name
                    + "/"
                    + os.path.basename(employee_name_pic_list[employee_name]),
                )
                create_employee_folder_and_copy_pic(
                    employee_id, employee_name, employee_name_pic_list[employee_name]
                )
                i += 1
                break

    workbook.save("employee_data.xlsx")
    logger.info(colored("数据已成功写入 employee_data.xlsx", "blue"))
