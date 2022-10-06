from typing import Optional, Union
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, PositiveInt, PositiveFloat

app = FastAPI(
    title="King County Housing Price Prediction",
    description="""
        Use simple Linear and Logistic Regression models to predict home attributes.
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
        model_path = 'models/bed_bath_regressor.pkl'
        features = np.array([[data.bedrooms, data.bathrooms]])
    else:
        model_path = 'models/full_regressor.pkl'
        features = np.array([[data.bedrooms, data.bathrooms, data.sqft]])

    model = joblib.load(model_path)
    return {'price': model.predict(features)[0]}

@app.post('/predict/logreg/')
async def predict_logreg(data: LogregData):
    features = np.array([[data.bedrooms, data.bathrooms, data.year]])

    model = joblib.load('models/logreg.pkl')
    return {'has_basement': str(model.predict(features)[0])}
