import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
import lancedb
import json
import pyarrow as pa
from typing import List, Dict, Any, Optional, Union
from lancedb.pydantic import Vector, LanceModel

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


class Poem:
    def __init__(
        self,
        id: str,
        author: Optional[str],
        title: Optional[str],
        paragraphs: List[str],
        colloquial_prose: List[str],
        embedded_colloquial_prose: List[List[float]],
    ):
        self.id = id
        self.author = author
        self.title = title
        self.paragraphs = paragraphs
        self.colloquial_prose = colloquial_prose
        self.embedded_colloquial_prose = embedded_colloquial_prose

    def __repr__(self):
        return (
            f"Poem(id={self.id}, author={self.author}, title={self.title}, "
            f"paragraphs={self.paragraphs}, colloquial_prose={self.colloquial_prose}, "
            f"embedded_colloquial_prose={self.embedded_colloquial_prose})"
        )


# 似乎有问题
def json_to_poem(json_data: Union[str, Dict]) -> Poem:
    # 如果输入是字符串，则解析为字典
    if isinstance(json_data, str):
        data = json.loads(json_data)
    # 如果输入已经是字典，则直接使用
    elif isinstance(json_data, dict):
        data = json_data
    else:
        raise ValueError("输入必须是 JSON 字符串或字典")

    return Poem(
        id=data["id"],
        author=data["author"],
        title=data["title"],
        paragraphs=data["paragraphs"],
        colloquial_prose=data["colloquial_prose"],
        embedded_colloquial_prose=data["embedded_colloquial_prose"],
    )


class PoemSchema(LanceModel):
    vector: Vector(1024)
    id: str
    author: Optional[str]
    title: Optional[str]
    paragraphs: List[str]
    colloquial_prose: List[str]


def create_poem_lancedb_table(
    uri: str,
    table_name: str,
    schema: Optional[Union[pa.Schema, LanceModel]],
    data: Optional[List[Dict[str, Any]]] = None,
) -> lancedb.table.Table:
    """
    创建并返回一个 LanceDB 表。

    参数:
    uri (str): LanceDB 的 URI。
    table_name (str): 要创建的表的名称。
    data (Optional[List[Dict[str, Any]]]): 包含初始数据的字典列表，默认为 None。

    返回:
    lancedb.table.Table: 创建的 LanceDB 表。
    """
    # 连接到 LanceDB
    db = lancedb.connect(uri=uri)

    # 创建表
    table = db.create_table(table_name, schema=schema, exist_ok=True)
    # 如果提供了数据，则插入数据
    if data is not None and len(data) > 0:
        table.add(data)

    return table


# 有问题
def jsonPoemFile2JsonList(json_file_path):
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            datas = json.load(file)
    except Exception as e:
        logger.error(colored(f"Error reading file {json_file_path}: {e}", "red"))
        datas = []

    for data in datas:
        Poem = json_to_poem(data)
        for i, vec in enumerate(Poem.embedded_colloquial_prose):
            pass


if __name__ == "__main__":
    """
    python test/usua3/try_lance_save.py
    """
    uri = "no_git_oic/poem-lancedb"
    json_file_path = "../chinese-poetry/trans_and_emb/全唐诗/poet.song.211000.json"
    name = "tang_song_poem"
    db = lancedb.connect(uri)
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            datas = json.load(file)
    except Exception as e:
        logger.error(colored(f"Error reading file {json_file_path}: {e}", "red"))
    logger.info(colored(f"{len(datas)}", "green"))

    tbl = create_poem_lancedb_table(uri, name, PoemSchema.to_arrow_schema())
    logger.info(colored(f"{db[name].head()}", "green"))
