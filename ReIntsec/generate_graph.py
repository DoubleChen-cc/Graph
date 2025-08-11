 
import networkx as nx
import numpy as np

# 参数设置
num_nodes = 100000
average_degree = 60

# 计算总边数
num_edges = num_nodes * average_degree // 2

# 使用配置模型生成符合幂律分布的度序列
degree_sequence = np.random.zipf(a=2.5, size=num_nodes)
degree_sequence = (degree_sequence / np.sum(degree_sequence) * num_edges * 2).astype(int)

# 确保度序列的和为偶数
if np.sum(degree_sequence) % 2 != 0:
    degree_sequence[0] += 1

# degree_sequence.sort()
# print(degree_sequence[:100])
# 生成图
G = nx.configuration_model(degree_sequence)
G = nx.Graph(G)  # 去除多重边和自环
G.remove_edges_from(nx.selfloop_edges(G))

# 保存图为txt文件
with open("data/txt_graph/random_graph_avrd60.txt", "w") as f:
    for edge in G.edges():
        f.write(f"{edge[0]}	{edge[1]}\n")
        f.write(f"{edge[1]}	{edge[0]}\n")