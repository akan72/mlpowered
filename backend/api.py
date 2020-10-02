from enum import Enum
from typing import Optional, List
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel

app = FastAPI(
    title="Zillow",
    description="""
    """
)

class Item(BaseModel):
    num_bedrooms: str
    num_bathrooms: float

class HouseType(str, Enum):
    str1 = '1'
    str2 = '2'
    str3 = '3'

@app.get('/')
async def hello(name: str = 'World'):
    return {'Hello': name}

@app.get('/items/{item_id}')
async def read_item(item_id: int, query: str = None):
    return {'item_id': item_id, 'query': query}

