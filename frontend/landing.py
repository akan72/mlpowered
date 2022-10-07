import os
import json
import pickle
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pydeck as pdk

import matplotlib.pyplot as plt
import requests

st.title("King County Housing Prices")
st.markdown(
"""
This is a demo of a Streamlit app explores the prices of houses sold
in King County, WA. between May of 2014 and May of 2015.

From this [Kaggle Competition](https://www.kaggle.com/harlfoxem/housesalesprediction).

Explore the distribution of houses sold across different cities within the county.
""")

@st.cache()
def load_data(path: str):
    data = pd.read_csv(path, index_col=[0])
    data = data.dropna(subset=['city'])
    data = data.drop(['id', 'sqft_living15', 'sqft_lot15'], axis=1)
    return data

data = load_data("../backend/data/processed/kc_housing_data_processed.csv")

top_cities = data['city'].value_counts()[:10].index.tolist()
top_cities = ['All'] + top_cities
display_city = st.selectbox(label='City Name', options=top_cities)

if display_city != 'All':
    data = data[data['city'] == display_city]

st.dataframe(data)

midpoint = (np.average(data["lat"]), np.average(data["long"]))
st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data[['lat', 'long']],
            get_position=["long", "lat"],
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))

hist_data = data[data['price'] < 2e6]
fig, ax = plt.subplots()

ax.hist(hist_data['price'], bins=50)
ax.set_title(f"Histogram of Housing Prices in {display_city if display_city != 'All' else 'King County'}")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

fig, ax = plt.subplots()
sns.histplot(data=hist_data, x='sqft_living', y='yr_built')
ax.set_title(f"Bivariate Histogram of Year Built and Living Space in {display_city if display_city != 'All' else 'King County'}")
ax.set_xlabel("Living Space (sqft)")
ax.set_ylabel("Year Built")
st.pyplot(fig)

st.subheader("Predict home price using (Bedrooms, Bathrooms, Sqft)")
bedrooms = st.slider('Bedrooms', 0, 10, 2, key='linreg_bed')
bathrooms = st.slider('Bathrooms', 0, 8, 2, key='linreg_bath')
sqft = st.slider('Square Feet', 200, 10000, 2000)

payload = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft': sqft,
}

if os.getenv('IS_IN_CONTAINER'):
    base_url = f"http://host.docker.internal:8000/"
else:
    base_url = "http://localhost:8000/"

endpoint = "predict/linreg/"

price_response = requests.post(base_url + endpoint, json=payload)
st.text(f"Predicted Price: ${price_response.json()['price']:.2f}")

intercept = price_response.json()['intercept']
coefficients = json.loads(price_response.json()['coefficients'])

st.markdown(f"""Linear Regression Equation\n: ```{intercept:.2f} + {coefficients[0]:.2f} * num_bedrooms + {coefficients[1]:.2f} * num_bathrooms + {coefficients[2]:.2f} * sqft```""")

fig = pickle.load(open('../backend/images/3dplot.pickle', 'rb'))
st.pyplot(fig)

st.markdown(f"Model Used: {price_response.json()['model']}")

st.subheader("Predict whether or not the home has a basement using (Bedrooms, Bathrooms, Year)")
bedrooms = st.slider('Bedrooms', 0, 10, 2, key='logreg_bed')
bathrooms = st.slider('Bathrooms', 0, 8, 2, key='logreg_bath')
year = st.slider('Year Built', 1950, 2015, 2000)

payload = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'year': year,
}

endpoint = "predict/logreg/"

basement_response = requests.post(base_url + endpoint, json=payload)
has_basement = basement_response.json()['has_basement']

basement_probability = basement_response.json()['basement_probability']

intercept = price_response.json()['intercept']
coefficients = json.loads(basement_response.json()['coefficients'])
st.markdown(f"""Logistic Regression Equation\n: ```{intercept:.2f} + {coefficients[0][0]:.2f} * num_bedrooms + {coefficients[0][1]:.2f} * num_bathrooms + {coefficients[0][2]:.4f} * year```""")

if has_basement:
    st.markdown(f"Logistic Regression predicted that the home has a basement with probability: {basement_probability * 100:.2f}%")
else:
    st.markdown(f"Logistic Regression predicted that the home does not have basement with probability: {(1-basement_probability) * 100:.2f}%")

st.markdown("## Logreg model performance")
st.image(Image.open('../backend/images/prcurve.png'))
st.image(Image.open('../backend/images/roccurve.png'))

st.text(f"Model Used: {basement_response.json()['model']}")

st.image(Image.open('../backend/images/swagger.png'))
