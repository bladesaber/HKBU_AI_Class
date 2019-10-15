import pandas as pd
import matplotlib.pyplot as plt
import re
import collections
import wordcloud

def show_null_examples(table):
    print('***** data null clean')
    print('*** data shape: ', table.shape)
    print(table.isnull().sum())

def process_dataset():
    table = pd.read_csv('D:\HKBU_AI_Classs\IT_Project\process.csv', index_col=0)
    for i in range(table.shape[0]):
        s = table['amazon_category_and_sub_category'][i]
        if isinstance(s, str):
            categories = s.replace(' ', '').split('>')
            category = categories[-1]
            table['amazon_category_and_sub_category'][i] = category
        else:
            continue
        print('%d finish' % i)
    table.to_csv('D:\HKBU_AI_Classs\IT_Project\dealer.csv')

# process_dataset()
table = pd.read_csv('D:\HKBU_AI_Classs\IT_Project\dealer.csv', index_col=0)

def major_manufacturer_market_and_customer_perentage():
    # market precentage
    df = table['manufacturer'].value_counts()
    df = df.head(10)
    df = pd.DataFrame(df)
    df.columns = ['count']
    df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table6.csv')

    df.plot(kind='bar', legend=True)
    plt.show()

    # -----------------------------------------------------------------------------------------------
    df2 = table.groupby('manufacturer')['number_of_reviews'].sum().sort_values(ascending=False)
    df2 = df2.head(10)
    df2 = pd.DataFrame(df2)

    df2.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table7.csv')

    df2.plot( kind='bar', legend=True)
    plt.show()

# major_manufacturer_market_and_customer_perentage()

head_5_manufacturer = ['LEGO', 'Disney', 'Playmobil', 'Star Wars', 'Mattel']

def major_manufacturer_categories_distribution():
    df = table[table['manufacturer'].isin(head_5_manufacturer)]
    df = df.groupby('manufacturer')['amazon_category_and_sub_category'].value_counts()

    df = pd.DataFrame(df)
    df.columns = ['count']
    df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table14.csv')

# major_manufacturer_categories_distribution()

def major_manufacturer_price_lecel():
    df = table[table['manufacturer'].isin(head_5_manufacturer)]
    df = df[(df['price'] > 0) & (df['price'] < 100)]

    df.groupby('manufacturer')['price'].plot(kind='kde', legend=True)
    plt.show()

    # --------------------------------------------------------------------------------
    # df = df.groupby('manufacturer')['price'].value_counts()
    # df = pd.DataFrame(df)
    # df.columns = ['count']
    # df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table8.csv')

# major_manufacturer_price_lecel()

def major_manufacturer_main_product():
    for cat in head_5_manufacturer:
        df = table[table['manufacturer'] == cat][['product_name', 'manufacturer', 'number_of_reviews']]
        df.sort_values(by=['number_of_reviews'], ascending=False, inplace=True)
        df = df.head(5)
        df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table9_%s.csv'%cat, index=False)

# major_manufacturer_main_product()

def major_brand():
    # ---------------------
    # df = pd.read_csv('D:\HKBU_AI_Classs\IT_Project\process.csv', index_col=0)
    # for i in range(df.shape[0]):
    #     s = df['amazon_category_and_sub_category'][i]
    #     if isinstance(s, str):
    #         if 'Characters & Brands' in s:
    #             categories = s.split('>')
    #             index = categories.index('Characters & Brands ')
    #             if index+1 <= len(categories):
    #                 df['amazon_category_and_sub_category'][i] = categories[index+1]
    #             else:
    #                 df['amazon_category_and_sub_category'][i] = 'no'
    #         else:
    #             df['amazon_category_and_sub_category'][i] = 'no'
    #     else:
    #         df['amazon_category_and_sub_category'][i] = 'no'
    #     print('%d finish' % i)
    #
    # df = df[df['amazon_category_and_sub_category']!='no'][['product_name', 'manufacturer', 'number_of_reviews', 'amazon_category_and_sub_category']]
    # df.to_csv('D:\HKBU_AI_Classs\IT_Project/brand.csv')

    # ----------------------
    df = pd.read_csv('D:\HKBU_AI_Classs\IT_Project/brand.csv', index_col=0)
    df = df.groupby('amazon_category_and_sub_category')['number_of_reviews'].sum().sort_values(ascending=False)
    df = pd.DataFrame(df.head(20))

    df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table10.csv')
    df.plot(kind='bar', legend=True)
    plt.show()

# major_brand()

def major_manufacturer_rating():
    df = table[table['manufacturer'].isin(head_5_manufacturer)]

    df.groupby('manufacturer')['average_review_rating'].plot(kind='kde', legend=True)
    plt.show()

    df = df.groupby('manufacturer')['average_review_rating'].value_counts()
    df = pd.DataFrame(df)
    df.columns = ['count']
    df.to_csv('D:\HKBU_AI_Classs\IT_Project\output_sample/table11.csv')

# major_manufacturer_rating()

def major_manufacture_question_wordcloud():
    remoced_list = ['does', 'the', 'what', 'how', 'this', 'to', 'for', 'and', 'it', 'a', 'she', 'he', 'can', 'of', 'in',
                    'you', 'is', 'do', 'with', 'have', 'as', 'are', 'would', 'will', 'when', 'or','',' ',]
    df = table[table['manufacturer'].isin(head_5_manufacturer)][['manufacturer', 'customer_questions_and_answers']]
    question_dict = {}
    for cat in head_5_manufacturer:
        question_dict[cat] = []
        for q_a in df[df['manufacturer'] == cat]['customer_questions_and_answers']:
            if isinstance(q_a, str):
                pairs = q_a.split('|')
                for pair in pairs:
                    question, answer = pair.split('//')
                    question = re.sub(r"\,|\!|\.|\?|\'|\"|\'|\*",' ', question)
                    question_dict[cat].extend(question.lower().split(' '))

        c = collections.Counter(question_dict[cat])

        for wd in remoced_list:
            c.pop(wd, '404')
        wc = wordcloud.WordCloud()
        wc.generate_from_frequencies(c)
        plt.imshow(wc)  # 显示词云
        plt.axis('off')  # 关闭坐标轴

        plt.imsave('D:\HKBU_AI_Classs\IT_Project\output_sample/table12_%s_question_wordcloud.png'%cat, wc)
        plt.show()  # 显示图像

# major_manufacture_question_wordcloud()
