import pandas as pd
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline


# 1. اختبار كلاس تجهيز البيانات (CustomData)
def test_custom_data_to_dataframe():
    data = CustomData(
        invoice="536365",
        stock_code="85123A",
        description="WHITE HANGING HEART T-LIGHT HOLDER",
        quantity=6,
        invoice_date="2026-08-01 08:26:00",
        price=2.55,
        customer_id="17850",
        country="United Kingdom",
    )

    df = data.get_data_as_data_frame()

    # أتأكد إن الخرج طالع Pandas DataFrame مش حاجة تانية
    assert isinstance(df, pd.DataFrame)
    # أتأكد إن الصفوف مش فاضية (فيها صف واحد على الأقل)
    assert len(df) == 1


# 2. اختبار كلاس التنبؤ (PredictionPipeline)
def test_prediction_pipeline_output():
    data = CustomData(
        invoice="536365",
        stock_code="85123A",
        description="WHITE HANGING HEART T-LIGHT HOLDER",
        quantity=6,
        invoice_date="2026-08-01 08:26:00",
        price=2.55,
        customer_id="17850",
        country="United Kingdom",
    )

    df = data.get_data_as_data_frame()
    pipeline = PredictionPipeline()

    prediction = pipeline.predict(df)

    # أتأكد إن الموديل طلع نتيجة (مش None)
    assert prediction is not None
    # أتأكد إن النتيجة فيها الأعمدة المتوقعة
    assert "Prediction" in prediction.columns
    assert "Cancel_Probability" in prediction.columns