import pickle
import io
from enum import Enum
from datetime import datetime
from typing import Optional, List, Annotated

from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class Fuel(str, Enum):
    diesel = 'Diesel'
    petrol = 'Petrol'
    cng = 'CNG'
    lpg = 'LPG'


class SellerType(str, Enum):
    individual = 'Individual'
    dealer = 'Dealer'
    trustmark_dealer = 'Trustmark Dealer'


class Transmission(str, Enum):
    manual = 'Manual'
    automatic = 'Automatic'


class Owner(str, Enum):
    first = 'First Owner'
    second = 'Second Owner'
    third = 'Third Owner'
    fourth = 'Fourth & Above Owner'
    test = 'Test Drive Car'


class Item(BaseModel):
    name: str
    year: int = Field(ge=1900, le=datetime.now().year)
    km_driven: int = Field(ge=0)
    fuel: Fuel
    seller_type: SellerType
    transmission: Transmission
    owner: Owner
    mileage: Optional[str] = Field(default=None)
    engine: Optional[str] = Field(default=None)
    max_power: Optional[str] = Field(default=None)
    torque: Optional[str] = Field(default=None)
    seats: float = Field(ge=2)


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_dict = item.model_dump()
    df = pd.DataFrame(item_dict, index=0)

    df = pd.DataFrame.from_dict(item_dict)
    pred = model.predict(df)

    return pred.round(2)


@app.post("/predict_items")
async def send_csv(file: Annotated[bytes, File()]):
    data = io.BytesIO(file)
    df = pd.read_csv(data)
    pred = model.predict(df)
    df['selling_price'] = pred
    csv_file = io.StringIO()
    df.to_csv(csv_file, index=False)

    csv_file.seek(0)

    response = StreamingResponse(
        iter([csv_file.getvalue()]),
        media_type='text/csv',
    )
    response.headers['Content-Disposition'] = 'attachment; filename="pred_selling_price.csv"'
    return response
