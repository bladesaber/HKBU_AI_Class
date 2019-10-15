import pandas as pd
import numpy as np
import re
import wordcloud
import collections
import matplotlib.pyplot as plt
import networkx as nx

# g = nx.Graph()
# g.add_node('a')
# g.add_node('a')
# g.add_node('b')
#
# nx.draw(g, with_labels=True)
# plt.show()

data = pd.DataFrame()
a = {"x": 'a', "y": 1, 'z':'h'}
data = data.append(a, ignore_index=True)
data = data.append({"x": 'b', "y": 1, 'z':'s'}, ignore_index=True)
data.loc[(data.x=='a') & (data.z=='h'), 'y'] = 100
print(data)