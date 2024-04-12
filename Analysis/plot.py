from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt

def test_seaborn():
    data = pd.read_csv('ZuCo_attn.csv', encoding='utf-8')
    # print(data['trt'])
    layer = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'layer 6',
             'layer 7', 'layer 8', 'layer 9', 'layer 10', 'layer 11', 'layer 12']
    eye = ['ffd', 'gd', 'gpt', 'trt', 'nFix']
    sent = data['sentence']
    print('=======ffd========')
    for lay in layer:
        att, ffd = [], []
        for i in range(len(sent)):
            if sent[i] != '[CLS]' and sent[i] != '[SEP]' and sent[i] != 0:
                att.append(data[lay][i])
                ffd.append(data['ffd'][i])
        r, p = spearmanr(att, ffd)
        print(r, p)
    print('========trt=========')
    for lay in layer:
        att, trt = [], []
        for i in range(len(sent)):
            if sent[i] != '[CLS]' and sent[i] != '[SEP]' and sent[i] != 0:
                att.append(data[lay][i])
                trt.append(data['trt'][i])
        r, p = spearmanr(att, trt)
        print(r, p)
    # r = 0.008703025102683665
    import seaborn as sns
    df = pd.DataFrame({'A': data["layer 12"],
                       'B': data["ffd"]})

    corr = df.corr(method='spearman')
    sns.heatmap(corr, annot=True)
    # plt.savefig('1.png')
    df = pd.DataFrame({"A": data["layer 12"], "B": data["ffd"]})
    sns.jointplot(x="A", y="B", data=df, alpha=0.2)
    # plt.savefig('2.png')

plt.show()
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
A = [...] # insert here your list of values for A
B = [...] # insert here your list of values for B
df = pd.DataFrame({'A': A,
                   'B': B})
corr = df.corr(method = 'spearman')
sns.heatmap(corr, annot = True)
plt.show()
'''

def plot_seaborn():
    # 导入本帖要用到的库，声明如下：
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import palettable
    from pandas import Series, DataFrame
    from sklearn import datasets
    import seaborn as sns

    # 导入鸢尾花iris数据集（方法一）
    # 该方法更有助于理解数据集
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    y_1 = np.array(['setosa' if i == 0 else 'versicolor' if i == 1 else 'virginica' for i in y])
    pd_iris = pd.DataFrame(np.hstack((x, y_1.reshape(150, 1))),
                           columns=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)',
                                    'class'])
    print(pd_iris['sepal length(cm)'])
    # astype修改pd_iris中数据类型object为float64
    pd_iris['sepal length(cm)'] = pd_iris['sepal length(cm)'].astype('float64')
    pd_iris['sepal width(cm)'] = pd_iris['sepal width(cm)'].astype('float64')
    pd_iris['petal length(cm)'] = pd_iris['petal length(cm)'].astype('float64')
    pd_iris['petal width(cm)'] = pd_iris['petal width(cm)'].astype('float64')
    print(pd_iris['sepal length(cm)'])
    # 导入鸢尾花iris数据集（方法二）
    # 该方法有时候会卡巴斯基，所以弃而不用
    # import seaborn as sns
    # iris_sns = sns.load_dataset("iris")
    sns.relplot(x='sepal length(cm)', y='sepal width(cm)', data=pd_iris,
                palette='Greens',
                height=7,
                style='class',
                hue='sepal length(cm)',  # 按照sepal length(cm)的长度显示marker的大小及颜色深浅，越长点越大，颜色越深
                size='sepal length(cm)',
                sizes=(50, 200),

                )
    sns.set(style="darkgrid", font_scale=1.5)

# plot_seaborn()
test_seaborn()