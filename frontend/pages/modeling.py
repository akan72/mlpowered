import os
import json
import pickle
import requests
import streamlit as st
from PIL import Image

st.title("King County Housing Prices")
st.subheader("Modeling")

st.markdown("#### Predict home price using (Bedrooms, Bathrooms, Sqft)")
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

st.markdown("#### Predict whether or not the home has a basement using (Bedrooms, Bathrooms, Year)")
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
