import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import datetime

# --- دالة تسجيل البيانات (Data Logging) للـ MLOps ---
def log_to_csv(df, pred, prob):
    os.makedirs("artifacts/logs", exist_ok=True)
    log_path = "artifacts/logs/inference_logs.csv"
    
    # نسخ الداتا للتسجيل
    log_df = df.copy()
    log_df['Prediction'] = pred[0]
    log_df['Probability'] = prob
    log_df['Timestamp'] = datetime.datetime.now()
    
    # لو الملف مش موجود نعمله بالـ Header، لو موجود نضيف من غير Header عشان الأعمدة تثبت
    if not os.path.exists(log_path):
        log_df.to_csv(log_path, index=False)
    else:
        log_df.to_csv(log_path, mode='a', header=False, index=False)

# إعدادات الصفحة
st.set_page_config(page_title="E-Retail Analytics", page_icon="📈", layout="wide")

# تحميل الموديل والـ Preprocessor
@st.cache_resource
def load_artifacts():
    model = joblib.load('artifacts/models/xgboost_model.pkl')
    preprocessor = joblib.load('artifacts/preprocessors/preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_artifacts()

st.title("🛒 E-Retail Cancellation Predictor")
st.markdown("---")

# المدخلات
st.sidebar.header("Input Features")
invoice = st.sidebar.text_input("Invoice Number", "536365")
stock_code = st.sidebar.text_input("Stock Code", "85123A")
quantity = st.sidebar.number_input("Quantity", value=6)
price = st.sidebar.number_input("Unit Price", value=2.55)
customer_id = st.sidebar.text_input("Customer ID", "17850")
country = st.sidebar.selectbox("Country", ["United Kingdom", "France", "Germany", "Others"])
invoice_date = st.sidebar.date_input("Invoice Date")

if st.sidebar.button("Analyze Order", use_container_width=True):
    with st.spinner('Processing...'):
        # 1. إنشاء DataFrame الأساسي
        df = pd.DataFrame([{
            'StockCode': stock_code, 'Price': price, 'Country': country,
            'Quantity': quantity, 'InvoiceDate': str(invoice_date),
            'Invoice': invoice, 'Customer ID': customer_id, 'Description': 'Item'
        }])
        
        # 2. بناء الميزات (Features) بترتيب مطابق تماماً لملف الـ reference
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Year"] = df["InvoiceDate"].dt.year
        df["Day_Name"] = df["InvoiceDate"].dt.day_name()
        df["Hour"] = 10
        df["Week"] = df["InvoiceDate"].dt.isocalendar().week
        df["Quarter"] = df["InvoiceDate"].dt.quarter
        df["Cancelled"] = df["Invoice"].astype(str).str.startswith("C").astype(int)
        
        df["Items_Count"] = 1
        df["Unique_Products"] = 1
        df["Average_Price"] = df["Price"]
        df["Customer_Orders"] = 1
        df["Customer_Frequency"] = 1
        df["Days_Since_Last_Order"] = 0
        df["Product_Avg_Price"] = df["Price"]
        df["Product_Frequency"] = 1
        df["Hour_Category"] = "Morning"
        df["Weekend"] = 0
        df["Season"] = "Summer"
        df["Is_UK"] = (df["Country"] == "United Kingdom").astype(int)
        
        df["Customer_Value"] = df["Price"]
        df["Total_Items_Bought"] = df["Quantity"]
        df["Customer_Lifetime"] = 0
        df["Items_Per_Order"] = df["Quantity"]
        df["Products_Per_Order"] = 1.0
        df["Product_Diversity"] = 1.0
        df["Avg_Items_Per_Product"] = 1.0
        df["Spend_Per_Order"] = df["Price"] * df["Quantity"]
        df["Spend_Per_Product"] = df["Price"]
        
        df["Price_Category"] = "Medium"
        df["Price_Log"] = np.log1p(df["Price"])
        df["Customer_Orders_Log"] = np.log1p(df["Customer_Orders"])
        df["Items_Count_Log"] = np.log1p(df["Items_Count"])
        df["Total_Items_Bought_Log"] = np.log1p(df["Total_Items_Bought"])
        df["Unique_Products_Log"] = np.log1p(df["Unique_Products"])
        df["Average_Price_Log"] = np.log1p(df["Average_Price"])
        df["Customer_Frequency_Log"] = np.log1p(df["Customer_Frequency"])
        df["Product_Frequency_Log"] = np.log1p(df["Product_Frequency"])
        df["Days_Since_Last_Order_Log"] = np.log1p(df["Days_Since_Last_Order"])
        df["Customer_Value_Log"] = np.log1p(df["Customer_Value"])
        df["Customer_Lifetime_Log"] = np.log1p(df["Customer_Lifetime"])
        
        df["High_Frequency_Customer"] = 0
        df["Expensive_Product"] = 0
        df["Loyal_Customer"] = 0
        df["New_Customer"] = 1
        df["Rare_Product"] = 0
        df["Popular_Product"] = 0
        
        df["Price_x_CustomerOrders"] = df["Price"]
        df["Price_x_ProductFreq"] = df["Price"]
        df["Orders_x_ProductFreq"] = 1
        df["Orders_x_Items"] = df["Quantity"]
        df["Items_x_Price"] = df["Price"]
        df["Items_x_ProductFreq"] = df["Quantity"]
        df["AveragePrice_x_ProductFreq"] = df["Price"]
        df["AveragePrice_x_Orders"] = df["Price"]
        df["Order_to_Product_Ratio"] = 1.0
        df["Price_to_ProductFreq"] = df["Price"]
        df["Frequency_to_Order"] = 1.0
        df["Price_to_Items"] = df["Price"]

        # إعادة ترتيب الأعمدة لتطابق تماماً ملف engineered_data.csv الأصلي
        reference_columns = [
            'StockCode', 'Price', 'Country', 'Cancelled', 'Year', 'Day_Name',
            'Hour', 'Week', 'Quarter', 'Items_Count', 'Unique_Products',
            'Average_Price', 'Customer_Orders', 'Customer_Frequency',
            'Days_Since_Last_Order', 'Product_Avg_Price', 'Product_Frequency',
            'Hour_Category', 'Weekend', 'Season', 'Is_UK', 'Customer_Value',
            'Total_Items_Bought', 'Customer_Lifetime', 'Items_Per_Order',
            'Products_Per_Order', 'Product_Diversity', 'Avg_Items_Per_Product',
            'Spend_Per_Order', 'Spend_Per_Product', 'Price_Category', 'Price_Log',
            'Customer_Orders_Log', 'Items_Count_Log', 'Total_Items_Bought_Log',
            'Unique_Products_Log', 'Average_Price_Log', 'Customer_Frequency_Log',
            'Product_Frequency_Log', 'Days_Since_Last_Order_Log',
            'Customer_Value_Log', 'Customer_Lifetime_Log',
            'High_Frequency_Customer', 'Expensive_Product', 'Loyal_Customer',
            'New_Customer', 'Rare_Product', 'Popular_Product',
            'Price_x_CustomerOrders', 'Price_x_ProductFreq', 'Orders_x_ProductFreq',
            'Orders_x_Items', 'Items_x_Price', 'Items_x_ProductFreq',
            'AveragePrice_x_ProductFreq', 'AveragePrice_x_Orders',
            'Order_to_Product_Ratio', 'Price_to_ProductFreq', 'Frequency_to_Order',
            'Price_to_Items'
        ]
        
        # التأكد من توفر جميع الأعمدة بالترتيب الصحيح للنموذج والتسجيل
        df_model_ready = df[reference_columns].copy()

        # 3. التوقع
        processed_data = preprocessor.transform(df_model_ready)
        pred = model.predict(processed_data)
        prob = model.predict_proba(processed_data)[0][1]

        # --- تسجيل البيانات للـ Monitoring (بنفس الأعمدة والترتيب) ---
        log_to_csv(df_model_ready, pred, prob)

        # 4. النتيجة
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        if pred[0] == 1:
            col1.error("Status: High Risk of Cancellation")
        else:
            col1.success("Status: Order Confirmed (Low Risk)")
        col2.metric("Confidence Score", f"{prob*100:.2f}%")
        st.progress(float(prob))