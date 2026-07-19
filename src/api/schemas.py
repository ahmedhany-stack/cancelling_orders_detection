from pydantic import BaseModel


class PredictionRequest(BaseModel):
    invoice: str
    stock_code: str
    description: str
    quantity: int
    invoice_date: str
    price: float
    customer_id: str
    country: str


class PredictionResponse(BaseModel):
    prediction: int
    cancel_probability: float