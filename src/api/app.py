from fastapi import FastAPI

from src.api.predict import router

app = FastAPI(
    title="Online Retail Cancellation Prediction API",
    version="1.0.0",
    description="Predict whether an order will be cancelled."
)

app.include_router(router)


@app.get("/")
def home():
    return {
        "message": "Online Retail Prediction API is Running"
    }