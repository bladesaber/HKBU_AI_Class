import pandas as pd
import numpy as np

#  设定 use_cols 是因为决策树只能用于离散特征
data = pd.read_csv('D:\HKBU_AI_Classs\DecisionTree\Bike_Sharing_Dataset\day.csv',
                   usecols=['season','holiday','weekday','workingday','weathersit','cnt'])
mean_data = np.mean(data.iloc[:, -1])    # 目标值的均值
# data.sample(5)

training_data = data.iloc[:int(0.7*len(data))].reset_index(drop=True)
testing_data = data.iloc[int(0.7*len(data)):].reset_index(drop=True)

# 这里使用方差来作为分裂的依据
def Var(data, f_name, y_name='cnt'):
    f_uni_val = np.unique(data.loc[:, f_name])

    # 对每一个可能的特征值做分裂测试并记录分裂后的加权方差
    f_var = 0
    for val in f_uni_val:
        # 把该特征等于某特定值的子集取出来
        cutset = data[data.loc[:, f_name] == val].reset_index()
        # 加权方差
        cur_var = (len(cutset)/len(data))*np.var(cutset.loc[:, y_name], ddof=1)
        f_var += cur_var

    return f_var

def RegTree(data, org_dataset, features, min_instances=5, y_name='cnt', p_node_mean=None):
    '''
    data：当前用于分裂的数据
    org_dataset：最原始的数据集
    '''
    # 如果数据量小于最小分割量，则直接输出均值
    if len(data) <= int(min_instances):
        return np.mean(data.loc[:, y_name])

    # 数据为空，返回父节点数据中的目标均值
    elif len(data) == 0:
        return np.mean(org_dataset.loc[:, y_name])

    # 无特征可分，返回父节点均值
    elif len(features) == 0:
        return p_node_mean

    else:
        # 当前节点的均值，会被传递给下层函数作为p_node_mean
        p_node_mean = np.mean(data.loc[:, y_name])

        # 找出最佳(方差最低)分裂特征
        f_vars = [Var(data, f) for f in features]
        best_f_idx = np.argmin(f_vars)
        best_f = features[best_f_idx]

        tree = {best_f: {}}

        # 移除已分裂的特征
        features = [f for f in features if f != best_f]

        # 以最佳特征的每一个取值划分数据并生成子树
        for val in np.unique(data.loc[:, best_f]):
            subset = data.where(data.loc[:, best_f] == val).dropna()
            tree[best_f][val] = RegTree(
                subset, data, features, min_instances, y_name, p_node_mean)

        return tree

def predict(query,tree,default=mean_data):
    '''
    query：一个测试样本，字典形式，{f:val,f:val,...}
    tree：生成树
    default：查找失败时返回的默认值，全样本的目标均值
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
    tree = RegTree(training_data, training_data, training_data.columns[:-1], 5)

    X_test = testing_data.iloc[:, :-1].to_dict(orient="records")
    Y_test = np.array(testing_data.iloc[:, -1])
    Y_pred = list()

    for item in X_test:
        Y_pred.append(predict(item, tree))
    Y_pred = np.array(Y_pred)

