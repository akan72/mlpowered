from typing import Optional, List, Union
import pickle
import numpy as np
import joblib

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
    bedrooms: PositiveInt
    bathrooms: Union[PositiveInt, PositiveFloat]
    year: PositiveInt

@app.get('/')
async def hello(name: str = 'World'):
    return {'Hello': name}

@app.post('/predict/linreg/')
async def predict_linreg(data: LinregData):
    if data.sqft is None:
        model = joblib.load('models/bed_bath_regressor.pkl')
        features = np.array([[data.bedrooms, data.bathrooms]])
    else:
        model = joblib.load('models/full_regressor.pkl')
        features = np.array([[data.bedrooms, data.bathrooms, data.sqft]])

    return {'price': model.predict(features)[0]}

@app.post('/predict/logreg/')
async def predict_logreg(data: LogregData):
    model = joblib.load('models/logreg.pkl')
    features = np.array([[data.bedrooms, data.bathrooms, data.year]])

    return {'has_basement': str(model.predict(features)[0])}
