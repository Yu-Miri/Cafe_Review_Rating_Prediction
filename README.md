# :coffee: Cafe Review Rating Predition

### 프로젝트 개요
- 기간 : 2023.01.23 ~ 2023.02.14
- 프로젝트 진행 인원 수 : 3명
- 주요 업무 및 상세 역할
  - 크롤링 : 카카오 맵의 카페 리뷰 및 평점 크롤링을 맡았습니다.
  - 데이터 전처리 : 한글, 공백을 제외한 문자 제거, 토큰화, 품사, 태깅, 불용어 사전 제작 후 삭제하는 전처리를 하였습니다.
  - 데이터 핸들링 및 모델 핸들링 : 데이터와 모델 간 task의 난이도를 파악한 후에 BOW의 하이퍼 파라미터를 통해 약 4,600개의 토큰에서 300개의 토큰으로 조절하여 데이터의 복잡도를 낮췄으며, 모델의 하이퍼 파라미터를 직접 바꾸는 경험을 해보았습니다.
  - 모델 평가 기준 선정 : 데이터의 분포 비대칭성이 존재하여 모델 평가 기준을 Recall으로 선정하였습니다.
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
  - Bow : 단어의 순서나 문맥을 고려하지 않으며, 문서나 문장을 단어 단위로 토큰화하여, 각 단어의 출현 빈도를 세어 벡터로 표현하는 방법
       <img width="192" alt="스크린샷 2023-06-04 오후 3 56 50" src="https://github.com/Yu-Miri/Mini_Project/assets/121469490/fa09d723-abee-40c4-8736-9c7ff3be17dc">
    
- 크롤링한 리뷰를 살펴 보았을 때 직접적으로 메뉴와 가격, 직원의 친절도에 대해 언급하면서 평가를 남긴 것을 확인할 수 있었으며, 이러한 평가로 카페에 대한 평점이 정해지기 때문에 카페에 대한 평가 표현의 빈도수에 따라 토큰의 개수가 적어져도 카페 평점 예측에 큰 영향이 없을 것이라 생각되어 전처리 이후에 Underfit 상태에서 데이터의 복잡도를 낮추기 위해 토큰의 개수를 줄였습니다.
- 토큰의 개수를 줄이고 모델의 hyper parameter tuning을 진행했을 때 Underfit이 해소되지 않은 것으로 보면, 모델이나 토큰 개수의 문제보다는 task의 난이도로 인한 모델 성능이 좋지 않다고 생각되어 다중 분류에서 이진 분류로 task의 난이도를 낮추어 개선하였습니다.

---
### 모델 학습 : LightGBM, Logistic Regression
- LightGBM[다중분류] -> **Underfit**

  - 데이터 핸들링 : (25244, 4630) ⇒ (25244, 300) [token 축소]
  - 모델 핸들링 : (max_depth = 3, n_estimators = 200) ⇒ (max_depth = 9, n_estimators = 400)
   <img width="281" alt="스크린샷 2023-06-04 오후 4 05 02" src="https://github.com/Yu-Miri/Mini_Project/assets/121469490/44c4c83a-f5a9-42dc-8cce-c96c38a5b86f">
  
  -> 핸들링 Before와 After를 비교해 보았을 때 오히려 더 낮아지는 score에 따라 심한 Underfit 상태로 판단했으며, 모델 성능의 문제보다는 치우쳐져 있는 평점 데이터 분포도의 한계를 발견했다. 이에 따라 휴리스틱 룰을 통해 다중 분류에서 이진 분류(Negative : 1점 - 3점, Positive : 4점 - 5점)로 새로운 가설을 생성하였다.
  -> 모델 hyper parameter tuning 라이브러리인 optuna 등을 사용해서 핸들링 했다면 성능을 더 높일 수 있지 않았을까 하는 아쉬움이 남는다.

- LightGBM[이진분류] -> **Underfit 해소**
  - Overfit 판단 : 복잡도가 낮은 Logistic Regression 모델 선정
   <img width="272" alt="스크린샷 2023-06-04 오후 4 06 20" src="https://github.com/Yu-Miri/Mini_Project/assets/121469490/5f21ae6f-e3fc-48b7-8056-0e1453863a7f">
  
  -> 휴리스틱 룰을 통해 이진 분류로 문제의 난이도를 낮춤으로써 모델의 성능을 개선할 수 있었지만, Train Recall score가 Test Recall score와 비교해 보았을 때 더 높은 것으로 Overfit 상태로 판단하였다.


- Logistic Regression -> **Overfit 해소**

   <img width="272" alt="스크린샷 2023-06-04 오후 4 25 19" src="https://github.com/Yu-Miri/Mini_Project/assets/121469490/3fc419f0-a06b-4496-8b65-38de56c1f08f">
  
  -> task의 난이도가 낮아지면서 Overfitting이 발생한 것으로 판단하였으며, 이를 해소하기 위해 모델을 복잡도가 낮은 logistic regression으로 교체한 결과 모델의 성능을 generalization 시킬 수 있었다.
---
Modeling 평가 기준 : Recall
- 수집한 Target 데이터의 분포 비대칭성 존재 → Accuracy 부적합 판단
- 실제 True인 것 중에서 모델이 True라고 예측한 비율
---

개선사항
- 치우쳐진 데이터의 분포도 : 리뷰 데이터 특성상 1점 - 3점 평점의 비중이 적어 평점 데이터 수집에 한계 존재
  ⇒ 휴리스틱 룰을 통해 이진 분류(Negative or Positive)로 새로운 가설 성립

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
    
