import os
from dotenv import load_dotenv
import logging
import sys
from neo4j import GraphDatabase


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
from test.rag.kg_stuff.com_rule_02_kg import parse_to_list


def test_connection(driver):
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Connection Successful!' AS message")
            for record in result:
                print(record["message"])  # 打印测试连接成功的消息
        print("Neo4j连接成功！")
    except Exception as e:
        print(f"连接失败: {e}")


# 定义导入函数
def import_data(driver, data):
    success_count = 0  # 计数器，记录插入成功的条数
    with driver.session() as session:
        for record in data:
            try:
                # 拼接 Cypher 查询，避免 relation 作为参数传入
                query = f"""
                MERGE (h:Entity {{name: $head}})  // 确保 head 节点唯一
                ON CREATE SET h.content = $content, h.id = $id  // 如果节点是新创建的，设置属性
                MERGE (t:Entity {{name: $tail}})  // 确保 tail 节点唯一
                MERGE (h)-[:`{record['relation']}`]->(t)  // 创建关系
                """

                # 执行拼接后的查询语句
                session.run(
                    query,
                    head=record["head"],
                    tail=record["tail"],
                    content=record.get("content", ""),  # 获取head对应的原文
                    id=record.get("id", ""),  # 获取head对应的id
                )
                success_count += 1  # 如果执行成功，计数器加1
            except Exception as e:
                print(f"插入数据失败: {e}")
    return success_count  # 返回插入成功的条数


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/kg_stuff/com_rule_03_db.py

    # 初始化Neo4j驱动
    uri = "neo4j://localhost:7687"  # 默认Neo4j地址
    username = "neo4j"  # 默认用户名
    password = "Tt123456!"  # 替换为你的密码

    kg_test = parse_to_list(open("no_git_oic/fsr_kg.txt", "r").read())
    received_kg_list = kg_test

    # 创建Neo4j连接驱动
    driver = GraphDatabase.driver(uri, auth=(username, password))
    # print(received_kg_list)

    # 测试连接
    test_connection(driver)
    # 导入数据
    suc_count = import_data(driver, received_kg_list)
    logger.info(f"成功条数:{suc_count}")
    """
MATCH (head:Entity {name: '公司培训管理制度'})-[r]->(tail)
RETURN head, r, tail
MATCH (head:Entity {name: '考勤管理制度'})-[r]->(tail)
RETURN head, r, tail

MATCH p=()-[r:`包括`]->() RETURN p

MATCH (n:Entity) RETURN n 

MATCH (n:Entity)
DETACH DELETE n
    """
