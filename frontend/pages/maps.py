import numpy as np
import streamlit as st
import pydeck as pdk

from utils import load_data

st.title("King County Housing Prices")
st.subheader("Mapping")

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
