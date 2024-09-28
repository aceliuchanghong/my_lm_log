import csv
import json
import os
import pandas as pd


def parquet_to_json(parquet_file, json_file):
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file)

    # 将 DataFrame 转换为 JSON 字符串并保存到文件中
    df.to_json(json_file, orient='records', lines=True, force_ascii=False)


def convert_csv_to_dict(csv_file, output_file=None):
    """
    将CSV文件中的数据转化为包含'dict'的列表，并生成JSON文件
    :param csv_file: CSV文件路径
    :param output_file: 输出的JSON文件路径，默认保存在data文件夹中，名字与原始CSV文件相同
    :return: 包含字典的列表
    """
    data_list = []

    # 读取CSV文件
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            # 对于每一行，创建一个新的字典
            data_dict = {
                "instruction": "a man and a women chatting",
                "input": row['He'],
                "output": row['She']
            }
            data_list.append(data_dict)

    # 如果没有指定输出文件路径，则默认保存在 data 文件夹中
    if output_file is None:
        base_name = os.path.basename(csv_file).replace('.csv', '.json')
        output_file = os.path.join('data', base_name)

    # 将列表写入到JSON文件
    with open(output_file, mode='w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

    return data_list


def add_input_field(json_file, output_file):
    with open(json_file, mode='r', encoding='utf-8') as file:
        json_list = json.load(file)  # 读取 JSON 文件数据
    updated_list = []
    for item in json_list:
        # 如果 input 字段不存在，添加 input 字段，值为空字符串
        if 'input' not in item:
            updated_item = {
                "instruction": item["instruction"],
                "input": "",  # 插入"input"字段
                "output": item["output"]
            }
        else:
            updated_item = {
                "instruction": item["instruction"],
                "output": item["output"]
            }
        updated_list.append(updated_item)

    with open(output_file, mode='w', encoding='utf-8') as file:
        json.dump(updated_list, file, ensure_ascii=False, indent=4)  # 写入 JSON 文件

    return updated_list


if __name__ == '__main__':
    # csv_file = 'data_row/sexy_chat_data_en_01.csv'
    # result = convert_csv_to_dict(csv_file)
    # print(result)
    # json_file = 'data_row/ruozhiba_qa.json'
    # output_file = 'data/ruozhiba_qa.json'
    # ans = add_input_field(json_file, output_file)
    # print(ans)
    parquet_file_path = 'data_row/sexual_label.parquet'
    json_file_path = 'data/train_01.json'
    parquet_to_json(parquet_file_path, json_file_path)
