import requests

url = 'http://localhost:8000'

linreg_url = url + '/predict/linreg/'

linreg_data_1 = {
    'bedrooms': 3,
    'bathrooms': 2,
}

response_1 = requests.post(linreg_url, json=linreg_data_1)
print(response_1.text)

linreg_data_2 = {
    'bedrooms': 3,
    'bathrooms': 2,
    'sqft': '2000'
}

response_2 = requests.post(linreg_url, json=linreg_data_2)
print(response_2.text)

logreg_data = {
    'bedrooms': 2,
    'bathrooms': 3,
    'year': 1950
}

logreg_url = url + '/predict/logreg/'
response_3 = requests.post(logreg_url, json=logreg_data)
print(response_3.text)
