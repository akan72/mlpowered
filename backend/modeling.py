from typing import List
import pandas as pd
import numpy as np
import joblib

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

""" Read in Kaggle data and add additional city and has_basement columns """

data = pd.read_csv('data/raw/kc_house_data.csv')
cities = pd.read_excel('data/processed/zipcode_city_mapping.xlsx')

data['city'] = data['zipcode'].map(
    dict(zip(cities['zipcode'], cities['city']))
)

data['has_basement'] = data['sqft_basement'].map(lambda x: x != 0)
data.to_csv('data/processed/kc_housing_data_processed.csv', index=False)

""" Train a simple model predicting price given # of bed/bathrooms """

features = ['bedrooms', 'bathrooms']

X = data[features]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .20, random_state=0
)

clf = LinearRegression()
clf.fit(X_train, y_train)

print('Linreg_1 Predictions:\n')
print(f"Linear Regression Equation\n: {clf.intercept_} + {clf.coef_[0]} * num_bedrooms + {clf.coef_[1]} * num_bathrooms\n")
print(clf.predict(np.array([[2, 2]])))
print(clf.predict(np.array([[4, 2]])))

joblib.dump(clf, 'models/bed_bath_regressor.pkl')

""" Train a second model predicting price given # of bed/bathrooms and sqft """

features = features + ['sqft_living']

X = data[features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .20, random_state=0
)

clf_2 = LinearRegression()
clf_2.fit(X_train, y_train)
joblib.dump(clf_2, 'models/full_regressor.pkl')

print('\nLinreg_2 Predictions:\n')
print(clf_2.predict(np.array([[2, 2, 1200]])))
print(clf_2.predict(np.array([[3, 4, 3500]])))

""" Train a Logistic Regression Model to predict has_basement """

features = ['bedrooms', 'bathrooms', 'yr_built']

X = data[features]
y = data['has_basement']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .20, random_state=0
)

clf_3 = LogisticRegressionCV(cv=5, random_state=0)
clf_3.fit(X_train, y_train)
joblib.dump(clf_3, 'models/logreg.pkl')

logreg_probabilities = clf_3.predict_proba(np.array([[2, 2, 1950]]))
print(f"\nLogreg Predictions:\n")
print(f"P(no_basement) = {logreg_probabilities[0][0]}, P(basement) = {logreg_probabilities[0][1]}")
print(clf_3.predict(np.array([[4, 4, 2010]])))
