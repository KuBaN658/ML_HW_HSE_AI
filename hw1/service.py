import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_dict = item.model_dump()
    for key, value in item_dict.items():
        item_dict[key] = [value]

    df = pd.DataFrame.from_dict(item_dict)
    pred = model.predict(df)

    return pred.round(2)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    item_dict = {}
    item = items[0]
    for key, value in item.model_dump().items():
        item_dict[key] = [value]

    for item in items[1:]:
        for key, value in item.model_dump().items():
            item_dict[key].append(value)

    df = pd.DataFrame.from_dict(item_dict)
    pred = model.predict(df)

    return pred.round(2).tolist()
