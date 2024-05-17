import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

level1_ratios = [0.01, 0.02, 0.05, 0.1]
df = pd.read_csv('/home/ubuntu/kevin/knowhere/build/2level_result.txt')
data = df.values.tolist()
# 假设CSV数据已经加载到这个列表中
# data = [
#     [0.2, 0.1, 0.80176, 0.0213003],
#     [0.2, 0.2, 0.91049, 0.0453816],
#     [0.2, 0.3, 0.95313, 0.0690793],
#     [0.2, 0.4, 0.97378, 0.0926763],
#     [0.2, 0.5, 0.98523, 0.115995],
# ]

# 提取search ratio为0.2的数据
for level1_ratio in level1_ratios:
    search_ratio_02_data = [row for row in data if row[0] == level1_ratio]

    # 提取search-data-size和recall的数据
    search_ratio_level2 = [row[1] for row in search_ratio_02_data]
    search_data_sizes = [row[3] for row in search_ratio_02_data]
    recalls = [row[2] for row in search_ratio_02_data]

    plt.figure(figsize=(10, 5))

    ax1 = plt.gca()
    ax1.set_xlabel('total search data size')
    ax1.set_ylabel('recall')
    ax1.plot(search_data_sizes, recalls, marker='o', color='r')

    ax2 = ax1.twiny()
    ax2.set_xlabel('level2 search data size')
    ax2.plot(search_ratio_level2, recalls, marker='^', color='b')

    # 绘制曲线
    # plt.plot(search_ratio_level2, recalls, marker='o')
    # plt.title(f'Recall vs Search Data Size')
    # plt.xlabel('Search Data Size')
    # plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig(f'recall_vs_search_data_size_{level1_ratio}.png')
    # plt.clf()