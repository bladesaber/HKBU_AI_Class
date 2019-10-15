import pandas as pd
import numpy as np

data = pd.read_csv('D:\HKBU_AI_Classs\DecisionTree\zoo_dtaset\zoo.data.csv', header=0, encoding="gbk")
# data=data.drop([0], axis=1)    # 首列是animal_name，丢弃
data = data.iloc[:, 1:]
# data.columns=range(len(data.columns))

def entropy(feature):
    '''
    :param feature: 一维分布
    '''
    uni_val, cnt = np.unique(feature, return_counts=True)    # 返回独特值与计数
    # 熵的计算
    H = np.sum([(-cnt[i]/np.sum(cnt))*np.log2(cnt[i]/np.sum(cnt)) for i in range(len(uni_val))])
    return H

def InfoGain(dataset, f_test_col, Y_col=-1):
    '''
    :param f_test_col: 需要划分feature的所在列
    :param Y_col: Lable所在的列
    :return:
    '''
    entropy_before = entropy(dataset.iloc[:, Y_col])    # 分割前的熵

    uni_val, cnt = np.unique(dataset.iloc[:, f_test_col],return_counts=True)    # 计算分割特征的独特值与计数
    # dropna() 过滤 dataset.iloc[:, f_test_col] != uni_val[i] 的行
    entropy_cond = np.sum([
        (cnt[i]/np.sum(cnt))*
        entropy(dataset.where(dataset.iloc[:, f_test_col] == uni_val[i]).dropna().iloc[:, Y_col]
                ) for i in range(len(uni_val))])
    return entropy_before - entropy_cond

def ID3(dataset, org_dataset, f_cols, Y_col=-1, p_node_cls=None):
    '''
    dataset: 用于分割的数据
    org_dataset: 用于计算优势类别的数据，父节点数据
    f_cols: 备选特征
    '''
    # 如果数据中的Y已经纯净了，则返回Y的取值
    # np.unique 在无return_counts=True下，只返回uni_val，这里指该节点下所有examples都为一类
    if len(np.unique(dataset.iloc[:, Y_col])) <= 1:
        return np.unique(dataset.iloc[:, Y_col])[0]

    # 如果传入数据为空(对应空叶节点)，则返回原始数据中数量较多的label值
    # 我倾向于用父节点的data中数量较多的label值
    elif len(dataset) == 0:
        uni_cls, cnt = np.unique(org_dataset.iloc[:, Y_col], return_counts=True)
        return uni_cls[np.argmax(cnt)]

    # 如果没有特征可用于划分，则返回父节点中数量较多的label值
    # 由于初始传入的是Index类型，所以这里不能用if not
    elif len(f_cols) == 0:
        return p_node_cls

    # 否则进行分裂
    else:
        # 得到当前节点中数量最多的label，递归时会赋给下层函数的p_node_cls
        cur_uni_cls, cnt = np.unique(dataset.iloc[:, Y_col], return_counts=True)
        # cur_node_cls 为当前节点下最多label
        cur_node_cls = cur_uni_cls[np.argmax(cnt)]
        # 释放内存
        del cur_uni_cls, cnt

        # 根据信息增益选出最佳分裂特征
        gains = [InfoGain(dataset, f_col, Y_col) for f_col in f_cols]
        best_f = f_cols[np.argmax(gains)]
        best_f_name = data.columns[best_f]

        # 更新备选特征 --》这里指保留的剩余特征
        f_cols = [col for col in f_cols if col != best_f]

        # 按最佳特征的不同取值，划分数据集并递归
        tree = {best_f_name: {}}
        for val in np.unique(dataset.iloc[:, best_f]):    # ID3对每一个取值都划分数据集
            sub_data = dataset.where(dataset.iloc[:, best_f] == val).dropna()
            tree[best_f_name][val] = ID3(sub_data, dataset, f_cols, Y_col, cur_node_cls)    # 分裂特征的某一取值，对应一颗子树或叶节点

        return tree

def predict(query, tree, default=-1):
    '''
    query：一个测试样本，字典形式，{f:val,f:val,...}
    tree：生成树
    default：查找失败时返回的默认类别
    '''
    for feature in list(query.keys()):
        if feature in list(tree.keys()):    # 如果该特征与根节点的划分特征相同
            try:
                sub_tree = tree[feature][query[feature]]    # 根据特征的取值来获取子节点

                if isinstance(sub_tree, dict):    # 判断是否还有子树
                    return predict(query, sub_tree)    # 有则继续查找
                else:
                    return sub_tree    # 是叶节点则返回结果
            except:    # 没有查到则说明是未见过的情况，只能返回default
                return default

def test():
    train_data = data.iloc[:80].reset_index(drop=True)
    test_data = data.iloc[80:].reset_index(drop=True)

    # 训练模型
    tree = ID3(train_data, train_data, list(range(train_data.shape[1] - 1)), -1)

    # DF转dict，一个条目为一个字典，返回一个字典的列表
    X_test = test_data.iloc[:, :-1].to_dict(orient="records")
    Y_test = list(test_data.iloc[:, -1])
    Y_pred = list()

    for item in X_test:
        Y_pred.append(predict(item, tree))

    print('acc:{}'.format(np.sum(np.array(Y_test) == np.array(Y_pred)) / len(Y_test)))

