from typing import Optional, List, Union
import pickle
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, PositiveInt, PositiveFloat

app = FastAPI(
    title="King County Housing Price Prediction",
    description="""
    """
)

class LinregData(BaseModel):
    bedrooms: PositiveInt
    bathrooms: Union[PositiveInt, PositiveFloat]
    sqft: Optional[PositiveInt]

class LogregData(BaseModel):
    price: float

@app.get('/')
async def hello(name: str = 'World'):
    return {'Hello': name}

@app.post('/predict/linreg/')
async def predict_linreg(data: LinregData):
    if data.sqft is None:
        model = pickle.load(open('models/bed_bath_regressor.pkl', 'rb'))
        features = np.array([[data.bedrooms, data.bathrooms]])

        return {'price': model.predict(features)[0]}

    else:
        model = pickle.load(open('models/full_regressor.pkl', 'rb'))
        features = np.array([[data.bedrooms, data.bathrooms, data.sqft]])

        return {'price': model.predict(features)[0]}
