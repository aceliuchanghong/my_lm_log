import networkx as nx

# https://github.com/rahulnyk/knowledge_graph/blob/main/extract_graph.ipynb
# Pyvis有一个内置的NetworkX助手，可以将我们的NetworkX图转换为PyVis对象。我们不需要再写代码
G = nx.Graph()
G.add_edge("A", "B", weight=4)
G.add_edge("B", "D", weight=2)
G.add_edge("A", "C", weight=3)
G.add_edge("C", "D", weight=4)
print(nx.shortest_path(G, "A", "D", weight="weight"))

nodes = [
    {"node_1": "玛丽", "node_2": "小羊", "edge": "属于"},
    {"node_1": "盘子", "node_2": "食物", "edge": "包含"},
]
for node in nodes:
    G.add_node(str(node))

for index, row in dfg.iterrows():
    G.add_edge(
        str(row["node_1"]), str(row["node_2"]), title=row["edge"], weight=row["count"]
    )
