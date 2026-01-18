import streamlit as st
import pandas as pd
import joblib
import os

# Paths
CLEANED_DATA = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
MODEL_PATH = r"D:\MLOps\models\sales_model.pkl"

st.set_page_config(page_title="MLOps Sales Dashboard", layout="wide")

st.title("💎 My Personal Sales AI")
st.markdown("---")

# --- SIDEBAR: Data Overview ---
st.sidebar.header("📊 Data Inventory")
if os.path.exists(CLEANED_DATA):
    df = pd.read_csv(CLEANED_DATA)
    st.sidebar.write(f"Total Rows: {len(df)}")
    st.sidebar.write(f"Categories: {', '.join(df['Category'].unique())}")
    
    # Show a small preview
    if st.sidebar.checkbox("Show Raw Data Preview"):
        st.sidebar.dataframe(df.head(10))

# --- MAIN CONTENT: The Prediction Station ---
st.header("🔮 AI Prediction Station")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Select Category")
    if os.path.exists(CLEANED_DATA):
        unique_cats = sorted(df['Category'].unique())
        user_choice = st.selectbox("Which category would you like to predict?", unique_cats)
        
        # Mapping (built to last)
        cat_map = {name: i for i, name in enumerate(unique_cats)}

with col2:
    st.subheader("AI Prediction Result")
    if st.button("Generate Prediction"):
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            code = [[cat_map[user_choice]]]
            prediction = model.predict(code)
            
            st.success(f"### Predicted Sales: ${prediction[0]:.2f}")
            st.metric(label="Expected Revenue", value=f"${prediction[0]:.2f}")
        else:
            st.error("Model not found! Please run the pipeline first.")

# --- VISUALIZATION SECTION ---
st.markdown("---")
st.header("📈 Business Trends")
if os.path.exists(CLEANED_DATA):
    # Dynamic Chart
    chart_data = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    st.bar_chart(chart_data)