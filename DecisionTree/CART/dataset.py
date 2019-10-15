import os
import pandas as pd
from collections import namedtuple

data_structure = namedtuple('data_structure', ['data', 'target'])

def load_breast_cancer():
    # path = os.path.join(os.path.dirname(__file__), 'breast-cancer-wisconsin', 'wdbc.data')
    df = pd.read_csv('D:\HKBU_AI_Classs\DecisionTree/breast-cancer-wisconsin\wdbc.csv', header=None)
    data = df.iloc[:, 2:].values
    target = df.iloc[:, 1].values
    return data_structure(data, target)