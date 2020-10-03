import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
import requests
import joblib

from sklearn.model_selection import train_test_split

from utils import plot_roc, plot_pr, evaluate

st.title("King County Housing Prices")
st.markdown(
"""
This is a demo of a Streamlit app explores the prices of houses sold
in King County, WA. between May of 2014 and May of 2015.

From this [Kaggle Competition](https://www.kaggle.com/harlfoxem/housesalesprediction).

Explore the distribution of houses sold across different cities within the county.
""")

st.set_option('deprecation.showPyplotGlobalUse', False)
url = 'http://localhost:8000/predict/linreg/'

@st.cache(persist=True)
def load_data(path: str):
    data = pd.read_csv(path)
    data = data.dropna(subset=['city'])
    data = data.drop(['id', 'sqft_living15', 'sqft_lot15'], axis=1)
    return data

data = load_data('../backend/data/processed/kc_housing_data_processed.csv')

top_cities = data['city'].value_counts()[:10].index.tolist()
top_cities = ['All'] + top_cities
display_city = st.selectbox(label='City Name', options=top_cities)

if display_city != 'All':
    data = data[data['city'] == display_city]

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

sns.jointplot(data=data, x='sqft_living', y='yr_built')
st.pyplot()

bedrooms = st.slider('Bedrooms', 0, 10, 2)
bathrooms = st.slider('Bathrooms', 0, 8, 2)
sqft = st.slider('Square Feet', 200, 10000, 2000)

json = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'yr_built': sqft
}

predicted_price = requests.post(url, json=json)
st.text(f"Predicted Price: {predicted_price.json()['price']}")

st.write(data)
