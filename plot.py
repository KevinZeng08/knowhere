import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_without_one_level():
    level1_ratios = [1]
    df = pd.read_csv('/home/ubuntu/kevin/knowhere/build/without_1level_result.txt')
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
        ax1.set_xlabel('level2 search data size')
        ax1.set_ylabel('recall')
        ax1.plot(search_ratio_level2, recalls, marker='o', color='r')

        # ax2 = ax1.twiny()
        # ax2.set_xlabel('level2 search data size')

        # 绘制曲线
        # plt.plot(search_ratio_level2, recalls, marker='o')
        # plt.title(f'Recall vs Search Data Size')
        # plt.xlabel('Search Data Size')
        # plt.ylabel('Recall')
        plt.grid(True)
        plt.savefig(f'recall_vs_search_data_size_{level1_ratio}.png')
        plt.clf()

    # total data size vs. recall
    # 提取level1_ratio=0.05，search-data-size和recall的数据
    markers = ['o', '^', 'd', 's', 'p']
    colors = ['r', 'b', 'g', 'y', 'm']
    plt.figure(figsize=(10, 5))
    for i, level1_ratio in enumerate(level1_ratios):
        data2 = [row for row in data if row[0] == level1_ratio]
        search_data_sizes = [row[3] for row in data2]
        recalls = [row[2] for row in data2]

        ax1 = plt.gca()
        ax1.set_xlabel('total data size')
        ax1.set_ylabel('recall')
        ax1.plot(search_data_sizes, recalls, marker=markers[i], color=colors[i])

    plt.legend([f'level1_ratio={level1_ratio}' for level1_ratio in level1_ratios])
    plt.grid(True)
    plt.savefig('recall_vs_total_data_size.png')

def plot_both():
    df_one_level = pd.read_csv('/home/ubuntu/kevin/knowhere/build/without_1level_result_128.txt')
    df_one_level_big = pd.read_csv('/home/ubuntu/kevin/knowhere/build/without_1level_result_1024.txt')
    df_two_level = pd.read_csv('/home/ubuntu/kevin/knowhere/build/2level_result.txt')
    df_two_level_big = pd.read_csv('/home/ubuntu/kevin/knowhere/build/2level_result_1024.txt')
    data_one_level = df_one_level.values.tolist()
    data_one_level_big = df_one_level_big.values.tolist()
    data_two_level = df_two_level.values.tolist()
    data_two_level_big = df_two_level_big.values.tolist()

    # plot two level
    best_ratio_two_level = 0.1
    two_level_data = [row for row in data_two_level if row[0] == best_ratio_two_level]
    search_data_sizes_two_level = [row[3] for row in two_level_data]
    recalls_two_level = [row[2] for row in two_level_data]

    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.set_xlabel('total data size')
    ax1.set_ylabel('recall')
    ax1.plot(search_data_sizes_two_level, recalls_two_level, marker='o', color='r')

    # plot two level big
    best_ratio_two_level_big = 0.1
    two_level_data_big = [row for row in data_two_level_big if row[0] == best_ratio_two_level_big]
    search_data_sizes_two_level_big = [row[3] for row in two_level_data_big]
    recalls_two_level_big = [row[2] for row in two_level_data_big]

    ax1.plot(search_data_sizes_two_level_big, recalls_two_level_big, marker='*', color='g')
    
    # plot one level
    one_level_data = [row for row in data_one_level if row[0] == 1]
    search_data_sizes_one_level = [row[3] for row in one_level_data]
    recalls_one_level = [row[2] for row in one_level_data]

    ax1.plot(search_data_sizes_one_level, recalls_one_level, marker='^', color='b')

    # plot one level big
    one_level_data_big = [row for row in data_one_level_big if row[0] == 1]
    search_data_sizes_one_level_big = [row[3] for row in one_level_data_big]
    recalls_one_level_big = [row[2] for row in one_level_data_big]

    ax1.plot(search_data_sizes_one_level_big, recalls_one_level_big, marker='d', color='y')

    plt.grid(True)
    plt.legend(['both level kmeans,num_cluster=128', 'both level kmeans,num_cluster=1024', 'second level kmeans,num_cluster=128', 'second level kmeans,num_cluster=1024'])

    # 图例

    plt.savefig('recall_two_level_vs_one_level.jpg')


def plot_three():
    df_first_level = pd.read_csv('/home/ubuntu/kevin/knowhere/build/1level_result_1024.txt')
    df_second_level = pd.read_csv('/home/ubuntu/kevin/knowhere/build/without_1level_result_128.txt')
    df_both_level = pd.read_csv('/home/ubuntu/kevin/knowhere/build/2level_result.txt')
    df_both_level_big = pd.read_csv('/home/ubuntu/kevin/knowhere/build/2level_result_1024.txt')

    data_first_level = df_first_level.values.tolist()
    data_second_level = df_second_level.values.tolist()
    data_both_level = df_both_level.values.tolist()
    data_both_level_big = df_both_level_big.values.tolist()

    # plot both level
    best_ratio_two_level = 0.1
    two_level_data = [row for row in data_both_level if row[0] == best_ratio_two_level]
    search_data_sizes_two_level = [row[3] for row in two_level_data]
    recalls_two_level = [row[2] for row in two_level_data]

    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.set_xlabel('total data size')
    ax1.set_ylabel('recall')
    ax1.plot(search_data_sizes_two_level, recalls_two_level, marker='o', color='r')

    # plot both level big
    best_ratio_two_level_big = 0.1
    two_level_data_big = [row for row in data_both_level_big if row[0] == best_ratio_two_level_big]
    search_data_sizes_two_level_big = [row[3] for row in two_level_data_big]
    recalls_two_level_big = [row[2] for row in two_level_data_big]

    ax1.plot(search_data_sizes_two_level_big, recalls_two_level_big, marker='*', color='g')

    # plot first level
    one_level_data = [row for row in data_first_level]
    search_data_sizes_one_level = [row[2] for row in one_level_data]
    recalls_one_level = [row[1] for row in one_level_data]

    ax1.plot(search_data_sizes_one_level, recalls_one_level, marker='^', color='b')

    # plot second level
    second_level_data = [row for row in data_second_level]
    search_data_sizes_second_level = [row[3] for row in second_level_data]
    recalls_second_level = [row[2] for row in second_level_data]

    ax1.plot(search_data_sizes_second_level, recalls_second_level, marker='d', color='y')

    plt.grid(True)
    plt.legend(['both level', 'both level, num_cluster=1024', 'first level', 'second level'])

    plt.savefig('recall_three_type.jpg')

# plot_both()
plot_three()