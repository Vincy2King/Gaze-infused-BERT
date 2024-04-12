import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
methods = ['FFD', 'GD', 'GPT', 'TRT', 'nFix']
datasets = ['Dataset1', 'Dataset2', 'Dataset3', 'Dataset4']

# 每个横坐标对应的四个数据集的数据
bert_base_data = np.array([
    [
        [2, 0, 6, 6],
        [1, 2, 6, 6],
        [0, 5, 8, 8],
        [0, 5, 4, 4]
    ],
    [
        [2, 0, 6, 6],
        [1, 2, 6, 6],
        [0, 5, 8, 8],
        [0, 4, 1, 1]
    ],
    [
        [2, 0, 6, 6],
        [1, 2, 6, 6],
        [0, 5, 8, 8],
        [0, 4, 5, 5]
    ],
    [
        [2, 0, 6, 6],
        [1, 2, 6, 6],
        [0, 5, 8, 8],
        [0, 4, 1, 1]
    ],
    [
        [2, 0, 6, 6],
        [1, 2, 6, 6],
        [0, 5, 8, 8],
        [0, 4, 5, 5]
    ],
])

bert_large_data = np.array([
    [
        [0, 10, 9, 9],
        [10, 0, 1, 1],
        [0, 10, 8, 8],
        [0, 10, 4, 4]
    ],
    [
        [0, 10, 9, 9],
        [10, 0, 11, 11],
        [0, 10, 9, 9],
        [0, 10, 4, 4]
    ],
    [
        [0, 10, 9, 9],
        [0, 10, 9, 9],
        [0, 10, 8, 8],
        [0, 10, 9, 9]
    ],
    [
        [0, 10, 9, 9],
        [0, 10, 9, 9],
        [0, 10, 8, 8],
        [0, 10, 9, 9]
    ],
    [
        [0, 10, 9, 9],
        [0, 10, 1, 1],
        [0, 10, 8, 8],
        [0, 10, 9, 9],
    ],
])

roberta_base_data = np.array([
    [
        [7, 0, 4, 4],
        [9, 6, 5, 6],
        [1, 8, 4, 8],
        [11, 1, 10, 10]
    ],
    [
        [7, 4, 10, 10],
        [6, 11, 9, 6],
        [1, 5, 8, 8],
        [11, 1, 6, 1]
    ],
    [
        [7, 4, 3, 3],
        [5, 6, 1, 6],
        [1, 5, 8, 8],
        [11, 1, 5, 5]
    ],
    [
        [7, 4, 9, 9],
        [1, 6, 11, 6],
        [1, 5, 8, 8],
        [11, 1, 10, 1]
    ],
    [
        [7, 4, 11, 11],
        [1, 6, 5, 5],
        [1, 5, 8, 8],
        [11, 1, 10, 10]
    ],
])

roberta_large_data = np.array([
    [
        [2, 0, 6, 2],
        [6, 11, 9, 6],
        [8, 2, 4, 8],
        [2, 0, 9, 2]
    ],
    [
        [2, 0, 6, 2],
        [6, 5, 2, 6],
        [8, 2, 4, 8],
        [2, 0, 5, 1]
    ],
    [
        [2, 7, 0, 0],
        [2, 9, 6, 6],
        [8, 2, 4, 8],
        [2, 0, 5, 5]
    ],
    [
        [2, 7, 6, 6],
        [2, 6, 0, 6],
        [8, 2, 4, 8],
        [2, 0, 5, 5]
    ],
    [
        [2, 7, 0, 0],
        [2, 0, 5, 5],
        [1, 4, 8, 8],
        [2, 0, 5, 5]
    ],
])

def draw(name):
    # 绘图
    if name=='bert_base':
        data = bert_base_data
    elif name == 'bert_large':
        data = bert_large_data
    elif name == 'roberta_base':
        data = roberta_base_data
    elif name == 'roberta_large':
        data = roberta_large_data
    else:
        print('wrong')
        exit()
    fig, ax = plt.subplots()

    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            print(data[i, j, :])
            if j==0:
                color='#A9D18E'
            elif j==1:
                color='#F4B183'
            elif j==2:
                color='#4FD1FF'
            else:
                color='#F58FE9'
            ax.scatter([i + 0.1 * j] * 4, data[i, j, :] + 1, c=color,label=f'{method}-{dataset}')

    ax.set_ylim(0, 13)
    # plt.legend()
    # plt.show()
    plt.savefig(name+'.jpg')

draw('bert_base')
draw('bert_large')
draw('roberta_base')
draw('roberta_large')