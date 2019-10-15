import numpy as np
from DecisionTree.CART.dataset import load_breast_cancer
from scipy import stats

data=load_breast_cancer()
X, Y = data.data, data.target
del data

def train_test_split(*arrays, test_size=0.25, random_state=None, shuffle=False):
    if not arrays or len(arrays) > 2:
        raise ValueError('params {} is illegal'.format(arrays))

    arr_1 = arrays[0]
    n_samples = len(arr_1)
    train_samples = int(n_samples * (1 - test_size))  # 训练样本数

    arr_2 = arrays[1] if len(arrays) > 1 else None
    if arr_2 is not None and n_samples != len(arr_2):
        raise ValueError('two arrays not equal')

    shuffle_idx = np.random.permutation(n_samples)
    if arr_2 is not None:
        return arr_1[shuffle_idx][:train_samples], \
               arr_1[shuffle_idx][train_samples:], \
               arr_2[shuffle_idx][:train_samples], \
               arr_2[shuffle_idx][train_samples:]
    else:
        return arr_1[shuffle_idx][:train_samples], arr_1[shuffle_idx][train_samples:]

X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.2)

# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

# 把X，Y拼起来便于操作
training_data=np.c_[X_train,Y_train]
testing_data=np.c_[X_test,Y_test]

# ----------------------------------------------------------------------------------------------------------------
def MSE(data,y_idx=-1):
    '''
    返回数据集目标值的MSE,计算均方差
    '''
    N=len(data)
    mean=np.mean(data[:,y_idx])
    return np.sum(np.square(data[:,y_idx]-mean))/N

def BinSplitData(data,f_idx,f_val):
    '''
    以指定特征与特征值二分数据集
    '''
    data_left=data[data[:,f_idx]<=f_val]
    data_right=data[data[:,f_idx]>f_val]
    return data_left,data_right

def Test(data, criteria='mse', min_samples_split=5, min_samples_leaf=5, min_impurity_decrease=0.0):
    '''
    对数据做test，找到最佳分割特征与特征值
    return: best_f_idx, best_f_val，前者为空时代表叶节点，两者都为空时说明无法分裂
    min_samples_split: 分裂所需的最小样本数，大于1
    min_samples_leaf: 叶子节点的最小样本数，大于0
    min_impurity_decrease: 分裂需要满足的最小增益
    '''
    n_sample, n_feature = data.shape

    # 数据量小于阈值则直接返回叶节点，数据已纯净也返回叶节点
    if n_sample < min_samples_split or len(np.unique(data[:,-1]))==1:
        return None, np.mean(data[:, -1])

    MSE_before = MSE(data)    # 分裂前的MSE
    best_gain = 0
    best_f_idx = None
    best_f_val = np.mean(data[:, -1])    # 默认分割值设为目标均值，当找不到分割点时返回该值作为叶节点

    # 遍历所有特征与特征值
    for f_idx in range(n_feature-1):
        for f_val in np.unique(data[:, f_idx]):
            data_left, data_right = BinSplitData(data, f_idx, f_val)    # 二分数据

            # 分割后的分支样本数小于阈值则放弃分裂
            if len(data_left) < min_samples_leaf or len(data_right) < min_samples_leaf:
                continue

            # 分割后的加权MSE
            MSE_after = len(data_left)/n_sample*MSE(data_left) + len(data_right)/n_sample*MSE(data_right)
            gain = MSE_before-MSE_after    # MSE的减小量为增益

            # 分裂后的增益小于阈值或小于目前最大增益则放弃分裂
            if gain < min_impurity_decrease or gain < best_gain:
                continue
            else:
                # 否则更新最大增益
                best_gain = gain
                best_f_idx, best_f_val = f_idx, f_val

    # 返回一个最佳分割特征与最佳分割点，注意会有空的情况
    return best_f_idx, best_f_val

def CART(data, criteria='mse', min_samples_split=5, min_samples_leaf=5, min_impurity_decrease=0.0):
    # 首先是做test，数据集的质量由Test函数来保证并提供反馈
    best_f_idx, best_f_val = Test(data, criteria, min_samples_split, min_samples_leaf, min_impurity_decrease)

    tree = {}
    tree['cut_f'] = best_f_idx
    tree['cut_val'] = best_f_val

    if best_f_idx == None:  # f_idx为空表示需要生成叶节点
        return best_f_val

    data_left, data_right = BinSplitData(data, best_f_idx, best_f_val)
    tree['left'] = CART(data_left, criteria, min_samples_split, min_samples_leaf, min_impurity_decrease)
    tree['right'] = CART(data_right, criteria, min_samples_split, min_samples_leaf, min_impurity_decrease)

    return tree


def predict_one(x_test, tree, default=-1):
    if isinstance(tree, dict):  # 非叶节点才做左右判断
        cut_f_idx, cut_val = tree['cut_f'], tree['cut_val']
        sub_tree = tree['left'] if x_test[cut_f_idx] <= cut_val else tree['right']
        return predict_one(x_test, sub_tree)
    else:  # 叶节点则直接返回值
        return tree

def predict(X_test, tree):
    return [predict_one(x_test, tree) for x_test in X_test]

# tree = CART(training_data)
# Y_pred = predict(X_test, tree)
# print('MSE:{}'.format(np.mean(np.square(Y_pred - Y_test))))
