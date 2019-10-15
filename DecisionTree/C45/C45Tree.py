from DecisionTree.C45.TreeNode import TreeNode
import pandas as pd
import numpy as np

class DecisionTreeC45:
    def __init__(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.root_node = TreeNode(parent=None, children=None, feature=None, category=None,
                                  X_data=self.X_train, Y_data=self.Y_train)
        self.features = self.get_features(self.X_train)
        self.tree_generate(self.root_node)

    def get_features(self, X_train_data):
        features = dict()
        for i in range(len(X_train_data.columns)):
            feature = X_train_data.columns[i]
            # .value_counts().keys() 只这个特征内含有的 unique 特征值
            features[feature] = list(X_train_data[feature].value_counts().keys())
        return features

    def tree_generate(self, tree_node):
        X_data = tree_node.X_data
        Y_data = tree_node.Y_data
        # get all features of the data set
        features = list(X_data.columns)

        # 如果Y_data中的实例属于同一类，则置为单结点，并将该类作为该结点的类
        if len(list(Y_data.value_counts())) == 1:
            tree_node.category = Y_data.iloc[0]
            tree_node.children = None
            return

        # 如果特征集为空，则置为单结点，并将Y_data中最大的类作为该结点的类
        elif len(features) == 0:
            tree_node.category = Y_data.value_counts(ascending=False).keys()[0]
            tree_node.children = None
            return

        # 否则，计算各特征的信息增益，选择信息增益最大的特征
        else:
            ent_d = self.compute_entropy(Y_data)

            XY_data = pd.concat([X_data, Y_data], axis=1)
            d_nums = XY_data.shape[0]
            max_gain_ratio = 0
            feature = None

            for i in range(len(features)):
                v = self.features.get(features[i])
                Ga = ent_d
                IV = 0
                for j in v:
                    dv = XY_data[XY_data[features[i]] == j]
                    dv_nums = dv.shape[0]
                    ent_dv = self.compute_entropy(dv[dv.columns[-1]])
                    if dv_nums == 0 or d_nums == 0:
                        continue
                    Ga -= dv_nums / d_nums * ent_dv
                    IV -= dv_nums/d_nums*np.log2(dv_nums/d_nums)

                if IV != 0.0 and (Ga/IV) > max_gain_ratio:
                    max_gain_ratio = Ga/IV
                    feature = features[i]
            # ----------------------------------------------------------------------------------------------------------
            # personal adjust
            # gainRations = [self.InfoGain(XY_data, f_test=feature) for feature in features]
            # feature = features[np.argmax(gainRations)]
            # max_gain_ratio = np.max(gainRations)

            # ----------------------------------------------------------------------------------------------------------
            # 如果当前特征的信息增益比小于阈值epsilon，则置为单结点，并将Y_data中最大的类作为该结点的类
            # 此处 0 为epsilon
            if max_gain_ratio < 0:
                tree_node.feature = None
                tree_node.category = Y_data.value_counts(ascending=False).keys()[0]
                tree_node.children = None
                return

            if feature is None:
                tree_node.feature = None
                tree_node.category = Y_data.value_counts(ascending=False).keys()[0]
                tree_node.children = None
                return
            tree_node.feature = feature

            # 否则，对当前特征的每一个可能取值，将Y_data分成子集，并将对应子集最大的类作为标记，构建子结点
            branches = self.features.get(feature)
            tree_node.children = dict()
            for i in range(len(branches)):
                X_data = XY_data[XY_data[feature] == branches[i]]
                if len(X_data) == 0:
                    category = XY_data[XY_data.columns[-1]].value_counts(ascending=False).keys()[0]
                    childNode = TreeNode(tree_node, None, None, category, None, None)
                    tree_node.children[branches[i]] = childNode
                    continue

                Y_data = X_data[X_data.columns[-1]]
                # 去除Label列
                X_data.drop(X_data.columns[-1], axis=1, inplace=True)
                # 去除该特征列
                X_data.drop(feature, axis=1, inplace=True)
                childNode = TreeNode(parent=tree_node, children=None, feature=None, category=None,
                                     X_data=X_data, Y_data=Y_data)
                tree_node.children[branches[i]] = childNode
                self.tree_generate(childNode)
            return

    # def compute_entropy(self, Y):
    #     ent = 0
    #     for cate in Y.value_counts(1):
    #         ent -= cate*np.log2(cate)
    #     return ent

    def compute_entropy(self, feature):
        '''
        :param feature: 一维分布
        '''
        uni_val, cnt = np.unique(feature, return_counts=True)  # 返回独特值与计数
        # 熵的计算
        H = np.sum([(-cnt[i] / np.sum(cnt)) * np.log2(cnt[i] / np.sum(cnt)) for i in range(len(uni_val))])
        return H

    def compute_GainRatio(self, feature, org_dataset_shape):
        gain = self.compute_entropy(feature)
        num = feature.shape[0]
        splitInformation = -num/org_dataset_shape[0] * np.log2(num/org_dataset_shape[0])
        return gain, splitInformation

    def InfoGain(self, dataset, f_test, Y_col=-1):
        '''
        :param f_test_col: 需要划分feature
        :param Y_col: Lable所在的列
        :return:
        '''
        entropy_before = self.compute_entropy(dataset.iloc[:, Y_col])  # 分割前的熵
        uni_val, cnt = np.unique(dataset.iloc[f_test], return_counts=True)  # 计算分割特征的独特值与计数

        gain_list,  splitInformation_list= [], []
        for i in range(len(uni_val)):
            dataset_v = dataset[dataset[f_test] == uni_val[i]].iloc[:, Y_col]
            gain, splitInformation = self.compute_GainRatio(dataset_v, dataset.shape)
            gain = (cnt[i] / np.sum(cnt)) * gain
            gain_list.append(gain)
            splitInformation_list.append(splitInformation)

        InfoGain_ration = (entropy_before - np.sum(gain_list))/np.sum(splitInformation_list)

        return InfoGain_ration
