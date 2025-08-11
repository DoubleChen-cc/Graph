#!/bin/bash

# 定义数据集数组
datasets=("email-Enron" "com-youtube" "mico" "orkut")
# 定义查询图数组
query_graphs=("2" "3" "4" "6" "11" "14" "15" "16")

# 外层循环遍历数据集
for dataset in "${datasets[@]}"
do
    # 内层循环遍历查询图
    for query_graph in "${query_graphs[@]}"
    do
        # 构建完整的命令并执行
        command="./build/cuts data/${dataset}.txt pattern/${query_graph}.txt"
        echo "Executing: $command"
        $command
    done
done
