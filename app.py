import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

st.set_page_config(page_title="EcomML Predictor", page_icon="📈", layout="centered")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

MODEL_PATH = os.path.join("models", "ridge_pipeline.joblib")
META_PATH = os.path.join("models", "ridge_metadata.json")

st.title("EcomML Total Amount Predictor")
st.caption("Predict purchase total using a trained Ridge pipeline")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found at models/ridge_pipeline.joblib. Please run the training script first.")
    st.stop()

model = load_model(MODEL_PATH)

# Load schema from metadata if available
schema = None
counts = {}
if os.path.exists(META_PATH):
    try:
        with open(META_PATH) as f:
            meta = json.load(f)
            schema = meta.get("input_schema")
            counts = (schema or {}).get("counts", {})
    except Exception:
        schema = None
        counts = {}

with st.sidebar:
    st.header("Inputs")

    qty_min = 1
    qty_max = 5
    if schema and "numeric" in schema and "Quantity" in schema["numeric"]:
        qty_min = int(schema["numeric"]["Quantity"].get("min", qty_min))
        qty_max = int(schema["numeric"]["Quantity"].get("max", qty_max))
    quantity = st.number_input("Quantity", min_value=qty_min, max_value=qty_max, value=min(max(qty_min, 3), qty_max))

    cat_vals = {
        "Product_Category": [
            "Beauty","Books","Electronics","Fashion","Food","Home & Garden","Sports","Toys"
        ],
        "Payment_Method": [
            "Bank Transfer","Cash on Delivery","Credit Card","Debit Card","Digital Wallet"
        ],
        "City": [
            "Adana","Ankara","Antalya","Bursa","Eskisehir","Gaziantep","Istanbul","Izmir","Kayseri","Konya"
        ],
    }
    if schema and "categorical" in schema:
        for k in cat_vals:
            if k in schema["categorical"]:
                cat_vals[k] = schema["categorical"][k]

    product_category = st.selectbox("Product Category", cat_vals["Product_Category"])
    payment_method = st.selectbox("Payment Method", cat_vals["Payment_Method"])
    city = st.selectbox("City", cat_vals["City"])

    # Show feature info
    with st.expander("Feature Info"):
        st.write({
            "Quantity_range": {"min": qty_min, "max": qty_max},
            "Product_Category_count": counts.get("Product_Category"),
            "Payment_Method_count": counts.get("Payment_Method"),
            "City_count": counts.get("City"),
        })

df = pd.DataFrame({
    "Quantity": [quantity],
    "Product_Category": [product_category],
    "Payment_Method": [payment_method],
    "City": [city],
})

st.subheader("Preview")
st.dataframe(df)

if st.button("Predict"):
    log_pred = model.predict(df)
    pred = float(np.expm1(log_pred)[0])
    st.success(f"Predicted Total Amount: {pred:,.2f}")

    if os.path.exists(META_PATH):
        meta = pd.read_json(META_PATH, typ="series")
        metrics = meta.get("metrics", {})
        with st.expander("Model Metrics"):
            st.json(metrics)
        with st.expander("Feature Importance"):
            fi = pd.DataFrame(meta.get("features", []))
            if not fi.empty:
                st.bar_chart(fi.set_index("feature")["importance"])