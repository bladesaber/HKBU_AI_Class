import numpy as np
import pandas as pd
import re
import traceback

regex = re.compile(r'co.uk\/(\S+)')

original = "D:\HKBU_AI_Classs\IT_Project/amazon_co-ecommerce_sample.csv"
process_path = 'D:\HKBU_AI_Classs\IT_Project\process.csv'

def drop_useless_col():
    print('***** data useless clean')
    table = pd.read_csv(original, index_col=None)
    table.drop(labels=[
        'uniq_id', 'number_available_in_stock', 'number_of_answered_questions', 'product_information', 'product_description'
    ], axis=1, inplace=True)
    print(table.columns)
    table.to_csv(process_path)

def drop_same_row_col():
    table = pd.read_csv(process_path, index_col=0)
    print('***** data duplicates clean')
    print('duplicates examples number: %d'%table.duplicated().sum())
    table.drop_duplicates(subset=None, keep='first')
    table.drop(labels=['description'], axis=1, inplace=True)
    print(table.columns)
    table.to_csv('D:\HKBU_AI_Classs\IT_Project\process.csv')

def show_null_examples():
    table = pd.read_csv(process_path, index_col=0)
    print('***** data null clean')
    print('*** data shape: ', table.shape)
    print(table.isnull().sum())
    # print(table.columns)

def handle_representation():
    table = pd.read_csv(process_path, index_col=0, encoding='utf-8')
    print('***** data ')

    for i in range(table.shape[0]):
        try:
            if isinstance(table['price'][i], str):
                if '-' in table['price'][i]:
                    a,b = table['price'][i].split('-')
                    a = np.float16(a.replace('-','').replace(' ','').replace('£', '').replace(',',''))
                    b = np.float16(b.replace('-','').replace(' ','').replace('£', '').replace(',',''))
                    table['price'][i] = a+b/2.0
                else:
                    table['price'][i] = np.float16(table['price'][i].replace('£', '').replace(' ','').replace(',',''))

            if isinstance(table['average_review_rating'][i], str):
                table['average_review_rating'][i] = np.float16(table['average_review_rating'][i].replace(' out of 5 stars','').replace(' ',''))
                # print('%d finish'%i)

            if isinstance(table['customers_who_bought_this_item_also_bought'][i], str):
                products = re.findall(regex, table['customers_who_bought_this_item_also_bought'][i])
                p = ''
                for k, product in enumerate(products):
                    product_s = product.split('/')[0]
                    if k>0:
                        p += '|'+product_s
                    else:
                        p += product_s
                table['customers_who_bought_this_item_also_bought'][i] = p
                # print('%d finish'%i)

            if isinstance(table['items_customers_buy_after_viewing_this_item'][i], str):
                products = re.findall(regex, table['items_customers_buy_after_viewing_this_item'][i])
                p = ''
                for k, product in enumerate(products):
                    product_s = product.split('/')[0]
                    if k>0:
                        p += '|'+product_s
                    else:
                        p += product_s
                # print(i,' : ',p)
                table['items_customers_buy_after_viewing_this_item'][i] = p
                # print('%d finish' % i)

            if isinstance(table['customer_reviews'][i], str):
                s = table['customer_reviews'][i].replace('\n', '')
                s = re.sub(r"\s{2,}", " ", s)
                a = s.split('//')
                p = ''
                for k in range(0, len(a), 4):
                    if k > len(a) - 1 or k + 1 > len(a) - 1:
                        break
                    if k > 0:
                        p += '|%s-%s' % (a[k].replace('|', ' '), a[k + 1].replace(' ', ''))
                    else:
                        p += '%s-%s' % (a[k].replace('|', ' '), a[k + 1].replace(' ', ''))
                table['customer_reviews'][i] = p
                # print('%d finish' % i)

            if isinstance(table['customer_questions_and_answers'][i], str):
                s = table['customer_questions_and_answers'][i].replace('\n', ' ')
                s = re.sub(r"\s{2,}", " ", s).replace('://','')

                p = ''
                a = s.split('|')
                for k, q_a in enumerate(a):
                    q, a = q_a.split('//')
                    if 'see more' in a:
                        r = re.findall('see more\s*(.*)\s*see less', s)[0]
                    else:
                        r = a
                    if k > 0:
                        p += '|%s//%s' % (q, r)
                    else:
                        p += '%s//%s' % (q, r)
                table['customer_questions_and_answers'][i] = p
                # print('%d finish' % i)

            if isinstance(table['number_of_reviews'][i], str):
                table['number_of_reviews'][i] = np.int16(table['number_of_reviews'][i].replace(',',''))
                # print('%d finish'%i)
            print('%d finish' % i)
            pass

        except Exception as e:
            print('str(e):\t\t', str(e))
            # print(table['price'][i],' - ',table['average_review_rating'][i])
            traceback.print_exc()
            break

    table.to_csv('D:\HKBU_AI_Classs\IT_Project\process.csv')

drop_useless_col()
drop_same_row_col()
# show_null_examples()
handle_representation()
