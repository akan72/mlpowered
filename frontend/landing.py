import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from utils import df_to_csv, load_data

st.title("King County Housing Prices")
st.subheader("Exploratory Data Analysis + Charting")
st.markdown(
"""
- This is an example of an interactive Data Application that performs Exploratory Data Analysis + Charting.
- The application also serves predictions from pre-trained ML models to a user in realtime.
- The primary [datasource](https://github.com/akan72/mlpowered/blob/main/backend/data/processed/kc_housing_data_processed.csv)
  originally comes from this [Kaggle competition](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). It contains
  price data from homes sold in King County, WA from May 2014 - May 2015.
- There is also a hand-curated [dataset](https://github.com/akan72/mlpowered/blob/main/backend/data/processed/zipcode_city_mapping.xlsx)
  of Zipcodes that is used to determine the city each house is in.
""")
data = load_data("../backend/data/processed/kc_housing_data_processed.csv")

top_cities = data['city'].value_counts()[:10].index.tolist()
top_cities = ['All'] + top_cities
display_city = st.selectbox(label='City Name', options=top_cities, help="Choose a city to view. All cities in the county by default")

if display_city != 'All':
    data = data[data['city'] == display_city]

st.dataframe(data)
st.download_button(
    label="Download data!",
    data=df_to_csv(data),
    file_name=f"{display_city.lower()}.csv",
    mime="text/csv"
)

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
