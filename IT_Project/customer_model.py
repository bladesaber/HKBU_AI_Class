import pandas as pd
import matplotlib.pyplot as plt
import collections
import wordcloud
import networkx as nx

def show_null_examples(table):
    print('***** data null clean')
    print('*** data shape: ', table.shape)
    print(table.isnull().sum())

table = pd.read_csv('D:\HKBU_AI_Classs\IT_Project\process.csv', index_col=0)
def purchase_count_price_level():
    df = table[(table['price']>0) & (table['price']<100)][['price', 'number_of_reviews']]
    df = df.groupby('price')['number_of_reviews'].sum()

    df = pd.DataFrame(df)
    df.columns = ['count']
    df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table13.csv')

# purchase_count_price_level()

head_5_manufacturer = ['LEGO', 'Disney', 'Playmobil', 'Star Wars', 'Mattel']
def comment_exploration():
    remoced_list = ['does', 'the', 'what', 'how', 'this', 'to', 'for', 'and', 'it', 'a', 'she', 'he', 'can', 'of', 'in',
                    'you', 'is', 'do', 'with', 'have', 'as', 'are', 'would', 'will', 'when', 'or',
                    '-5.0', '-4.0', '-3.0', '-2.0', '-1.0', 'has', 'so', 'my', 'was', 'not', 'his',
                    'i', 'all', 'very', 'that', 'her', 'which', 'love', 'on', 'but', 'good', 'great',
                    'bought', 'year', 'get', 'set', 'got', 'be', 'one', 'if', 'some', 'really', 'up',
                    'at', 'toy', 'loved', 'like', 'five', 'out', 'much', 'also', "it's", 'they', 'their',
                    'these', 'only', 'who', 'two', 'from', 'we', 'been', 'then', 'just', 'two', 'than',
                    'also', 'your', 'about', 'had', 'its', 'loves', 'there', 'more', 'buy', 'excellent',
                    'nice', 'well', 'bit', 'by', 'other', 'value', 'it.', 'build', 'made', 'put', 'them',
                    'our', 'an', 'were', 'both', 'took', 'take', 'brilliant', 'any', 'lot', 'into', 'me',
                    'hours', 'every', 'lots', 'too', 'last', 'play', 'played', "don't", 'after', 'four',
                    '5', 'bits', 'think', 'better', 'product', 'building', 'day', 'around', 'comes', 'still',
                    'could', 'sets', 'being', 'thing', 'most', 'off', 'toys', 'make', 'item', 'thought', 'go',
                    'looks', 'pieces', 'no', 'even', '2', 'it', 'happy', 'years', 'different', 'now', 'arrived',
                    '',' ',]

    def parse_string(s):
        p = ''
        if isinstance(s, str):
            comments = s.lower().split('|')
            for comment in comments:
                p += ' '+comment
        return p

    df = table[table['manufacturer'].isin(head_5_manufacturer)]['customer_reviews']
    df = df.apply(parse_string)
    p = []
    for i in range(df.shape[0]):
        p.extend(df.iloc[i].split(' '))

    c = collections.Counter(p)
    for wd in remoced_list:
        c.pop(wd, '404')

    wc = wordcloud.WordCloud()
    wc.generate_from_frequencies(c)
    plt.imshow(wc)  # 显示词云
    plt.axis('off')  # 关闭坐标轴

    plt.imsave('D:\HKBU_AI_Classs\IT_Project\output_sample/table15.png', wc)

    plt.show()  # 显示图像

comment_exploration()

def market_structure():
    def mapcategories(srs):
        if pd.isnull(srs):
            return []
        else:
            return [cat.strip() for cat in srs.split(">")]

    df = table['amazon_category_and_sub_category'].apply(mapcategories)

    node_weight_dict = pd.DataFrame(columns=['node', 'weight'])
    edge_weight_dict = pd.DataFrame(columns=['source', 'target', 'weight'])
    for i in range(df.shape[0]):
        if len(df.iloc[i])>0:
            for cat in df.iloc[i]:
                if cat in node_weight_dict['node'].values:
                    node_weight_dict.loc[node_weight_dict.node==cat, 'weight'] += 1
                else:
                    node_weight_dict = node_weight_dict.append({'node':cat, 'weight':1}, ignore_index=True)

            for k in range(len(df.iloc[i]) - 1):
                original = df.iloc[i][k]
                end = df.iloc[i][k+1]

                if (original in edge_weight_dict['source'].values) and (end in edge_weight_dict['target'].values):
                    edge_weight_dict.loc[(edge_weight_dict.source==original) & (edge_weight_dict.target==end), 'weight'] += 1
                elif (original in edge_weight_dict['target'].values) and (end in edge_weight_dict['source'].values):
                    edge_weight_dict.loc[(edge_weight_dict.target==original) & (edge_weight_dict.source==end), 'weight'] += 1
                else:
                    edge_weight_dict = edge_weight_dict.append({'source':original, 'target':end, 'weight':1}, ignore_index=True)

        print('%d finish'%i)

    # print(node_weight_dict)
    # print(edge_weight_dict)

    node_weight_dict.to_csv('D:\HKBU_AI_Classs\IT_Project/node_weight.csv')
    edge_weight_dict.to_csv('D:\HKBU_AI_Classs\IT_Project/edge_weight.csv')

    # DG = nx.DiGraph()
    # df.map(lambda cats: DG.add_nodes_from(cats))
    # df.map(lambda cats: [DG.add_edge(cats[i], cats[i + 1]) for i in range(len(cats) - 1)])

# market_structure()
