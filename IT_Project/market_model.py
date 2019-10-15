import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def show_null_examples(table):
    print('***** data null clean')
    print('*** data shape: ', table.shape)
    print(table.isnull().sum())

def create_market_dataset():
    table = pd.read_csv('D:\HKBU_AI_Classs\IT_Project\process.csv', index_col=0)
    df = table[['product_name', 'amazon_category_and_sub_category', 'number_of_reviews', 'price']]
    df.to_csv('D:\HKBU_AI_Classs\IT_Project\market.csv')

def process_dataset(table):
    for i in range(table.shape[0]):
        s = table['amazon_category_and_sub_category'][i]
        if isinstance(s, str):
            categories = s.replace(' ','').split('>')
            category = categories[-1]
            table['amazon_category_and_sub_category'][i] = category
        else:
            continue
        print('%d finish'%i)
    table.to_csv('D:\HKBU_AI_Classs\IT_Project\market.csv')

# create_market_dataset()
table = pd.read_csv('D:\HKBU_AI_Classs\IT_Project\market.csv', index_col=0)
# process_dataset(table)
# show_null_examples(table)

def category_split_and_market_percentage():
    # df = table['amazon_category_and_sub_category'].value_counts()
    df = table['amazon_category_and_sub_category'].value_counts()
    df = df.head(5)

    print('head 5 ********')
    print(df)
    df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table1.csv', header=['count'])

    # df.plot(kind='bar', legend=True)
    # plt.show()

# category_split_and_market_percentage()

head_5_category = ['Vehicles', 'Toys', 'ScienceFiction&Fantasy', 'BeadArt&Jewellery-Making', 'Packs&Sets ']
def category_price_distribution():
    # flash print
    df = table[table['amazon_category_and_sub_category'].isin(head_5_category)]
    df = df[(df['price']>0) & (df['price']<100)]
    df.groupby('amazon_category_and_sub_category')['price'].plot(kind='kde', legend=True)
    plt.show()

    # for cat in head_5_category:
    #     cat_df = table[table['amazon_category_and_sub_category']==cat]
    #     cat_df = cat_df[(cat_df['price']>0) & (cat_df['price']<100)]
    #     cat_count = pd.value_counts(values=cat_df['price'])
    #     print(cat_count)

    # --------------------------------------------------------------------------------
    # df = table[table['amazon_category_and_sub_category'].isin(head_5_category)]
    # df = df[(df['price']>0) & (df['price']<100)]
    # df = df.groupby('amazon_category_and_sub_category')['price'].value_counts()
    #
    # df = pd.DataFrame(df)
    # df.columns = ['count']
    # df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table2.csv')

    # df.plot(kind='bar', legend=True)
    # plt.show()

# category_price_distribution()

def most_popular_category():
    df = table.groupby('amazon_category_and_sub_category')['number_of_reviews'].sum().sort_values(ascending=False)
    # print(df.head(20))
    df = pd.DataFrame(df.head(20))
    df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table3.csv')

    df.plot(kind='bar', legend=True)
    plt.show()

def most_popular_product():
    df = table.sort_values(by='number_of_reviews', ascending=False)[['product_name','number_of_reviews']]
    # print(df.head(20))

    df = df.head(20)
    # df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table4.csv', index=False)

    df.plot(x='product_name', y='number_of_reviews', kind='bar', legend=True)
    plt.show()

# category_split_and_market_percentage()
# category_price_distribution()
# most_popular_category()
# most_popular_product()

def categories_network():
    ''' fail, need to cut edge'''
    products = pd.read_csv('D:\HKBU_AI_Classs\IT_Project\process.csv', index_col=0)
    def mapcategories(srs):
        if pd.isnull(srs):
            return []
        else:
            return [cat.strip() for cat in srs.split(">")]

    category_lists = products['amazon_category_and_sub_category'].apply(mapcategories)
    category_lists.map(lambda lst: len(lst)).value_counts()

    DG = nx.DiGraph()

    category_lists.map(lambda cats: DG.add_nodes_from(cats))
    category_lists.map(lambda cats: [DG.add_edge(cats[i], cats[i + 1]) for i in range(len(cats) - 1)])

    print("The number of categorical links possible is {0}.".format(len(DG.edges())))

    nx.draw(DG, with_labels=True)
    plt.show()

categories_network()
