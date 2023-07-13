import re
import numpy as np
from konlpy.tag import Okt

# 노이즈 제거를 함수화한다.
def preprocess_okt(reviews):
    okt = Okt()

    # 1. 한글, 공백을 제외한 문자를 제거한다.
    review_text = re.sub("[^가-힣\s]", "", reviews)

    # 2. okt 객체를 활용하여 형태소 토큰화, 품사 태깅한다.
    word_review = okt.pos(review_text, stem=True)

    # 3. 노이즈, 불용어를 제거한다.
    word_review = [(token, pos) for token, pos in word_review if not token in stopwords and len(token) > 1]

    # 4. 명사, 동사, 형용사를 제외한 품사를 제거한다.
    word_review = [token for token, pos in word_review if pos in ['Noun', 'Verb', 'Adjective']]

    # 전처리, Join 후 비어있으면 NaN값으로 변경한다.
    if word_review:
        return ' '.join(word_review)
    else:
        return np.nan