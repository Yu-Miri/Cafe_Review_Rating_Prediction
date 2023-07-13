from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import recall_score

def model_dataset(reviews, bow_train):
    X_train, X_test, y_train, y_test = train_test_split(bow_train,reviews.Score_div,
                                                        stratify=reviews.Score_div,
                                                        train_size=0.85, test_size=0.15,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def modeling_LGBM(X_train, X_test, y_train, y_test):
    lgbm_model = LGBMClassifier(application = 'binary', boosting_type = 'gbdt', learning_rate = 0.4,
                                max_depth=5, n_estimators = 400)
    
    lgbm_model.fit(X_train, y_train)

    train_pred = lgbm_model.predict(X_train)
    test_pred = lgbm_model.predict(X_test)
    
    print('Train Recall Score : ', recall_score(y_train, train_pred, average='macro'))
    print('Test Recall Score : ', recall_score(y_test, test_pred, average='macro'))

def modeling_Logistic(X_train, X_test, y_train, y_test):
    logistic_model = LogisticRegressionCV(random_state=0)
    
    logistic_model.fit(X_train, y_train)

    train_pred = logistic_model.predict(X_train)
    test_pred = logistic_model.predict(X_test)

    print('Train Recall Score : ', recall_score(y_train, train_pred, average='macro'))
    print('Test Recall Score : ', recall_score(y_test, test_pred, average='macro'))
