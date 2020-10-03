from typing import List
import pandas as pd
import numpy as np
import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.decomposition import PCA

from utils import plot_roc, plot_pr, evaluate

""" Read in Data and perform initial train/test split """
data = pd.read_csv('data/raw/kc_house_data.csv')

cities = pd.read_excel('data/processed/zipcode_city_mapping.xlsx')
data['city'] = data['zipcode'].map(
    dict(zip(cities['zipcode'], cities['city']))
)

features: List[str] = ['bedrooms', 'bathrooms']

X: pd.DataFrame = data[features]
y: pd.DataFrame = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .20, random_state=0
)

""" Train a simple model predicting price given # of bed/bathrooms """
clf = LinearRegression()

clf.fit(X_train, y_train)

print(clf.coef_, clf.intercept_)
print(clf.predict(np.array([[2, 2]])))
print(clf.predict(np.array([[4, 2]])))

bed_bath_regressor = 'models/bed_bath_regressor.pkl'
with open(bed_bath_regressor, 'wb') as f:
    pickle.dump(clf, f)

""" Train a second model predicting price given # of bed/bathrooms and sqft """
features = features + ['sqft_living']

X = data[features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .20, random_state=10
)

clf_2 = LinearRegression()

clf_2.fit(X_train, y_train)

print(clf_2.coef_, clf_2.intercept_)
print(clf_2.predict(np.array([[2, 2, 1200]])))
print(clf_2.predict(np.array([[3, 4, 3500]])))

full_regressor = 'models/full_regressor.pkl'
with open(full_regressor, 'wb') as f:
    pickle.dump(clf_2, f)

processed_data_url = 'data/processed/kc_housing_data_processed.csv'
data.to_csv(processed_data_url)
