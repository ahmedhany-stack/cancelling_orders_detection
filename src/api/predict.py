from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictionRequest, PredictionResponse
from src.pipeline.prediction_pipeline import (
    PredictionPipeline,
    CustomData,
)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


@router.post("/", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    try:

        custom_data = CustomData(
            invoice=data.invoice,
            stock_code=data.stock_code,
            description=data.description,
            quantity=data.quantity,
            invoice_date=data.invoice_date,
            price=data.price,
            customer_id=data.customer_id,
            country=data.country,
        )

        df = custom_data.get_data_as_data_frame()

        pipeline = PredictionPipeline()

        prediction = pipeline.predict(df)

        return PredictionResponse(
            prediction=int(prediction.iloc[0]["Prediction"]),
            cancel_probability=float(
                prediction.iloc[0]["Cancel_Probability"]
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )