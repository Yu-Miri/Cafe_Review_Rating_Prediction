# :coffee: Cafe Review Rating Predition

### 프로젝트 개요
- 기간 : 2023.01.23 ~ 2023.02.14
- 프로젝트 진행 인원 수 : 3명
- 주요 업무 및 상세 역할
    - 크롤링 : Selenium을 사용하여 카카오 맵의 카페 리뷰 및 평점 크롤링을 맡았습니다.
    - 데이터 전처리 : Sklearn 라이브러리를 사용하여 한글, 공백을 제외한 문자를 제거하고, 토큰화, 품사 태깅,  리뷰를 하나씩 확인해 보며 학습에 악영향을 미칠 것이라 판단되는 불용어를 선정하여 불용어 사전 제작 후 삭제하는 전처리를 하였습니다.
    - 데이터 및 모델 핸들링 : 데이터와 모델 간 task의 난이도를 파악한 후에 BOW의 하이퍼 파라미터를 통해 약 4,600개의 토큰에서 300개의 토큰으로 조절하여 데이터의 복잡도를 낮췄으며, 데이터의 한계로 다중 분류에서 이진 분류로 학습 난이도를 낮추어 성능을 개선하고자 하였습니다. 이에 더하여, 모델의 복잡도에 따라 달라지는 성능을 확인하며, Task 난이도에 맞는 모델을 적용하여 성능을 개선하였습니다.
    - 모델 평가 기준 선정 : 데이터의 분포 비대칭성이 존재하였으며, 이에 따라 Positive인 경우에 높게 측정되는 Accuracy로 평가하는 것은 부적합하다고 판단하여 모델 평가 기준을 실제 Positive인 것 중에 모델이 맞춘 확률을 측정하는 Recall으로 선정하였습니다.
- 사용 언어 및 개발 환경 : Google colab Pro+, Python3.8, BeautifulSoup, Selenium, Sklearn, Tensorflow
---
### 문제 정의
- 대부분의 소비자들은 가고 싶은 카페를 선정할 때 평점과 리뷰를 참고하지만, 각 플랫폼 별 평점이 다르고, 광고성 리뷰 또는 악의성 리뷰로 인하여 어떤 플랫폼의 어떤 리뷰가 믿을만한 리뷰인지에 대한 혼란을 겪게 된다.

     <img width="600" alt="스크린샷 2023-06-04 오후 3 45 33" src="https://github.com/Yu-Miri/Mini_Project/assets/121469490/09e7531f-e3ce-443e-97c8-0c838601c9e3">
     <img width="600" alt="스크린샷 2023-06-04 오후 3 46 06" src="https://github.com/Yu-Miri/Mini_Project/assets/121469490/dcef931b-46fb-4d25-984d-8789a0baed39">


### 해결 방안
<프로젝트 목적>
- Text Mining을 통해 리뷰 데이터에서 평점에 영향을 미치는 정보를 추출하고, Sklearn 라이브러리를 활용하여 카페의 리뷰에 대해서 일관성 있는 평점을 예측할 수 있도록 모델링하는 것을 경험한다.

<프로젝트 내용>
- 팀 프로젝트를 통해 여러 플랫폼의 리뷰와 평점을 분석하여 과장된 광고성 리뷰나 악의성 리뷰로부터 벗어나 리뷰와 평점을 참고해 카페를 선정하는 소비자들의 의사결정에 도움이 될 수 있는 카페 리뷰 평점의 평균 지표를 제공한다.
---

### 데이터 설명
  <img width="311" alt="스크린샷 2023-06-04 오후 3 51 43" src="https://github.com/Yu-Miri/Mini_Project/assets/121469490/a70dd95b-3eb5-4eee-a8fe-fbd17851948f">

- 출처 : Kakao Map, Dining Code
- Data Size : Kakao Map(14,000) + Dining Code(11,000) = 25,000개
- Location : 카페 밀집도가 많은 서울시 행정구역
- Feature : Review
- Target : 특정 카페의 리뷰를 바탕으로 예측한 총 평점

---
### 데이터 전처리
- 한글, 공백을 제외한 문자 제거 : 이모티콘, 특수문자 등 의미 없는 문자 존재
- Okt 객체를 이용하여 형태소 토큰화, 품사 태깅 : KoNLPy
- 명사, 동사, 형용사를 제외한 품사 제거 : 실질적 의미를 담고 있는 품사 선정
- 노이즈, 불용어 제거 : 평점에 영향을 미치지 않는 단어 기준으로 불용어 선정
- Vectorization
    - TF-IDF : 단어 출현 빈도와 문서 빈도 수를 통한 희귀성을 고려한 단어의 중요성 수치화 기법
        
        ![스크린샷 2023-05-29 오전 11.17.16.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/71af51d7-23ac-44a5-a7e2-1951ebcd2cd8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-29_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.17.16.png)
        
    - 크롤링한 리뷰를 살펴 보았을 때 직접적으로 메뉴와 가격, 직원의 친절도에 대해 언급하면서 평가를 남긴 것을 확인할 수 있었으며, 이러한 평가로 카페에 대한 평점이 정해지기 때문에 카페에 대한 평가 표현의 빈도수에 따라 토큰의 개수가 적어져도 카페 평점 예측에 큰 영향이 없을 것이라 생각되어 전처리 이후에 Underfit 상태에서 데이터의 복잡도를 낮추기 위해 토큰의 개수를 줄였습니다.
    - 토큰의 개수를 줄이고 모델의 hyper parameter tuning을 진행했을 때 Underfit이 해소되지 않은 것으로 보면, 모델이나 토큰 개수의 문제보다는 task의 난이도로 인한 모델 성능이 좋지 않다고 생각되어 다중 분류에서 이진 분류로 task의 난이도를 낮추어 개선하였습니다.

---
### 모델 학습

- Model : **LightGBM, Logistic Regression**
    
    
    ![스크린샷 2023-05-29 오후 2.12.21.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a83f610c-53d5-4a8f-85fc-3d72cfb83b7c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-29_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.12.21.png)
    
    ![스크린샷 2023-05-29 오후 2.59.06.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b07def8a-7a08-4e25-9f3d-a8dfb36dc745/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-29_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.59.06.png)
    
    ![스크린샷 2023-05-29 오후 2.59.40.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/486c3848-a9c8-4066-9767-c1b1188731a8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-29_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.59.40.png)
    
    LightGBM [다중분류]
    
    - 데이터 핸들링 : (25244, 4630) ⇒ (25244, 300) [token 축소]
    - 모델 핸들링 : (max_depth = 3, n_estimators = 200) ⇒ (max_depth = 9, n_estimators = 400)
    - Light GBM Train Recall Score 그래프의 Before와 After를 비교해 보았을 때 오히려 더 낮아지는 score에 따라 Underfit 상태로 판단했으며, 모델 성능의 문제보다는 치우쳐져 있는 평점 데이터 분포도에서의 문제로, 여러 Class에서 이진으로 분류하도록 1점에서 3점의 평점은 Negative, 4점에서 5점의 평점은  Positive로 하여 새로운 가설을 생성하였다.
        - 모델 hyper parameter tuning 라이브러리인 optuna 등을 사용해서 핸들링 했다면 성능을 더 높일 수 있지 않았을까 하는 아쉬움이 남는다.
    
    Logistic Regression [이진분류]
    
    - task의 난이도가 낮아지면서 모델의 성능은 올라갔지만, Train Recall score가 Test Recall score와 비교해 보았을 때 더 높았기에 Overfit 상태라 판단하였다.
    - 이를 해소하기 위해 모델을 복잡도가 낮은 Logistic Regression으로 교체한 결과 모델의 성능을 일반화할 수 있었다.

- Modeling 평가 기준
    - Recall : 실제 True인 것 중에 모델이 실제로 맞춘 비율을 말한다. Recall = True Positive / (True Positive + False Negative)
    - 수집한 Target 데이터의 분포 비대칭성 존재하였으며, Positive의 Label이 상대적으로 많았기에 양성인 경우에 높게 측정되는 Accuracy로 평가하는 것은 부적합하다고 판단하였다.

### 개선사항

- 치우쳐진 데이터의 분포도 : 리뷰 데이터 특성상 1점 - 3점 평점의 비중이 적어 평점 데이터 수집에 한계 존재

           ⇒ 이진 분류(Negative or Positive)로 새로운 가설 성립

- 네이버 지도 리뷰는 평점이 없고 리뷰만 존재했기에 데이터 사용 불가능
- 구글 리뷰는 크롤링 난이도와 시간적 여유가 없었기에 사용 불가능하여 다이닝 코드로 대체
- 다중 분류를 위해 1점 - 3점 평점의 더 많은 데이터를 확보해 Label 분포 균등화
- 다양한 플랫폼들의 리뷰 크롤링

-----------
## Installation

### Requirements
- Python==3.8

      git clone https://github.com/Yu-Miri/Mini_Project.git
      cd Mini_Project/text_mining
      pip install konlpy
    

### Preparing for DataFrame

      import pandas as pd
      from df_processing import df_process
    
      reviews = pd.read_csv('reviews.csv', index_col=0, encoding='utf-8-sig')
      reviews, bow_train = df_process(reviews)
 
 
### Modeling[[LGBM, Logistic]]
    
      from modeling import model_dataset, modeling_LGBM, modeling_Logistic
    
      X_train, X_test, y_train, y_test = model_dataset(reviews, bow_train)
      modeling_LGBM(X_train, X_test, y_train, y_test)
      modeling_Logistic(X_train, X_test, y_train, y_test)
 
 
### Predict
Recommended procedure : Requirements -> Preparing for DataFrame -> Modeling -> Predict

      reviews['pred'] = lgbm_model.predict(X_train)
    
