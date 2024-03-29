import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

GLOBAL_RANDOM_STATE = 0
GLOBAL_TEST_SIZE = .20

""" Read in Kaggle data and add additional city and has_basement columns """

data = pd.read_csv("data/raw/kc_house_data.csv")
cities = pd.read_excel("data/processed/zipcode_city_mapping.xlsx")

data['city'] = data['zipcode'].map(
    dict(zip(cities['zipcode'], cities['city']))
)

data['has_basement'] = data['sqft_basement'].map(lambda x: x != 0)
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date').sort_index()

data.to_csv(f'data/processed/kc_housing_data_processed.csv')

""" Train a simple model predicting price given # of bed/bathrooms """

features = ['bedrooms', 'bathrooms']

X = data[features].values
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=GLOBAL_TEST_SIZE, random_state=GLOBAL_RANDOM_STATE
)

clf = LinearRegression()
clf.fit(X_train, y_train)

print('Linreg_1 Predictions:\n')
print(f"Linear Regression Equation\n: {clf.intercept_} + \
     {clf.coef_[0]} * num_bedrooms + {clf.coef_[1]} * num_bathrooms\n")
print(clf.predict(np.array([[2, 2]])))
print(clf.predict(np.array([[4, 2]])))

y_pred = clf.predict(X_test)
joblib.dump(clf, f"models/bed_bath_regressor.pkl")

# Create a 3d plot showing the predicted plane of the 2d regression
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test)

X, Y = np.meshgrid(X_test[:, 0], X_test[:, 1])
Z = clf.intercept_ + clf.coef_[0] * X + clf.coef_[1] * Y

ax.plot_surface(X, Y, Z, color='r', alpha=0.5)
ax.set_xlabel('Number of Bedrooms')
ax.set_ylabel('Number of Bathrooms')
ax.set_zlabel('Home Price')

# Save the image as a pickle file for consumption in the frontend
pickle.dump(fig, open('images/3dplot.pickle', 'wb'))

""" Train a second model predicting price given # of bed/bathrooms and sqft """

features = features + ['sqft_living']

X = data[features].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=GLOBAL_TEST_SIZE, random_state=GLOBAL_RANDOM_STATE
)

clf_2 = LinearRegression()
clf_2.fit(X_train, y_train)
joblib.dump(clf_2, f"models/full_regressor.pkl")

print('\nLinreg_2 Predictions:\n')
print(clf_2.predict(np.array([[2, 2, 1200]])))
print(clf_2.predict(np.array([[3, 4, 3500]])))

""" Train a Logistic Regression Model to predict has_basement """

features = ['bedrooms', 'bathrooms', 'yr_built']

X = data[features].values
y = data['has_basement']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=GLOBAL_TEST_SIZE, random_state=GLOBAL_RANDOM_STATE
)

clf_3 = LogisticRegressionCV(cv=5, random_state=GLOBAL_RANDOM_STATE)
clf_3.fit(X_train, y_train)
joblib.dump(clf_3, f"models/logreg.pkl")

# Save the ROC and PR curves for the logreg model
RocCurveDisplay.from_estimator(clf_3, X_test, y_test)
plt.savefig('images/roccurve.png')

PrecisionRecallDisplay.from_estimator(clf_3, X_test, y_test)
plt.savefig('images/prcurve.png')

logreg_probabilities = clf_3.predict_proba(np.array([[2, 2, 1950]]))
print(f"\nLogreg Predictions:\n")
print(f"P(no_basement) = {logreg_probabilities[0][0]}, P(basement) = {logreg_probabilities[0][1]}")
print(clf_3.predict(np.array([[4, 4, 2010]])))
