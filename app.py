# app.py
import streamlit as st
import pandas as pd
from src.train import train_model
from src.predict import predict_sales

st.title("📊 Sales Prediction App")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Predict Sales", "About"])

# --- Train Model Page ---
if page == "Train Model":
    st.header("🔧 Train the Model")
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, metrics = train_model()
        st.success("✅ Model trained successfully!")
        st.subheader("📊 Model Performance")
        for k, v in metrics.items():
            st.write(f"**{k}:** {v:.4f}")

# --- Predict Sales Page ---
elif page == "Predict Sales":
    st.header("💡 Predict Sales from Ad Spend")
    tv = st.number_input("TV Advertising Budget", min_value=0.0, value=100.0)
    radio = st.number_input("Radio Advertising Budget", min_value=0.0, value=20.0)
    newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, value=30.0)

    if st.button("Predict"):
        input_data = {"TV": tv, "Radio": radio, "Newspaper": newspaper}
        price = predict_sales(input_data)
        st.success(f"📈 Predicted Sales: **{price} units**")

# --- About Page ---
elif page == "About":
    st.header("ℹ️ About this Project")
    st.markdown("""
    This project predicts **Sales** based on advertising spend on **TV, Radio, and Newspaper**.  
    It uses a **Random Forest Regression model** trained on the Advertising dataset.  
    """)
