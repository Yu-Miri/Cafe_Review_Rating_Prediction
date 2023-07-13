import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
from okt_preprocessing import preprocess_okt as pre_okt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

'''

review DataFrame < columns = Review, Score >을 준비한다.
Dtype : Review = object, Score = int 64, Score div = object, preprocessed = object

'''
reviews = pd.read_csv('reviews.csv', index_col=0, encoding='utf-8-sig')

def df_process(reviews):
    '''
    input = csv 파일 이름
    '''
    
    reviews = reviews.drop_duplicates()
    reviews = reviews.dropna()
    reviews = reviews.reset_index(drop=True)

    # 2진 분류 (필요 시 실행)
    reviews.loc[reviews['Score'] > 3, 'Score div'] = 'Positive'
    reviews.loc[reviews['Score'] < 4, 'Score div'] = 'Negative'

    # 불용어 사전을 준비한다.
    stopword = open('stop_words.txt', 'rt').read()
    stopwords = stopword.split(' ')

    # 전처리 함수를 DataFrame에 적용시킨다.
    tqdm.pandas()
    reviews['preprocessed'] =(reviews['Review'.progress_apply(lambda x : pre_okt(x))])

    # 전처리 후 Nan 값을 제거한다.
    reviews = reviews.dropna()
    reviews = reviews.reset_index(drop = True)

    # 한 문장으로 join된 전처리 후의 문장들을 공백 기준으로 분리한다. 
    def my_tokenizer(text):
        return text.split()

    # BOW를 통해서 count 희소 표현을 생성하여 토큰 수 행렬을 변환한다.
    bow_vect = CountVectorizer(tokenizer=my_tokenizer, min_df=5, max_df=0.8, max_features=300) 
    bow_train = bow_vect.fit_transform(raw_documents = reviews['preprocessed'])
    print('토큰의 수 : ', len(bow_vect.get_feature_names_out()))
    
    return reviews, bow_train