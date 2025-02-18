# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# ตั้งค่า layout ของหน้าเว็บ
st.set_page_config(page_title="Health & Financial ML/NN Project", layout="wide")

# ----- Custom CSS สำหรับตกแต่งหน้าเว็บ -----
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
    }
    .css-1outpf7 {
        background-color: #ffffff;
        border-bottom: 2px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----- ฟังก์ชันช่วยโหลดข้อมูล CSV -----
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# ----- ฟังก์ชันสำหรับเตรียมข้อมูล Health Data (Machine Learning) -----
def prepare_health_dataset(df):
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['HeartRate'] = df['HeartRate'].fillna(df['HeartRate'].mean())
    return df

# ----- ฟังก์ชันสำหรับเตรียมข้อมูล Financial Data (Neural Network) -----
def prepare_financial_dataset(df):
    df['StockPrice'].fillna(df['StockPrice'].mean(), inplace=True)
    df['Income'].fillna(df['Income'].mean(), inplace=True)
    df['Expense'].fillna(df['Expense'].mean(), inplace=True)
    return df

# ----- ฟังก์ชันสำหรับโหลดโมเดล Neural Network (สมมุติว่าฝึกไว้แล้ว) -----
@st.cache_resource
def load_nn_model():
    model_path = "models/financial_nn.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = None
    return model

# ----- สร้างแท็บสำหรับ Navigation -----
tabs = st.tabs(["Overview", "Machine Learning", "Neural Network", "Demo Machine Learning", "Demo Neural Network"])

# ==========================================================
# Tab 1: Overview
# ==========================================================
with tabs[0]:
    st.title("Overview ของโปรเจค")
    st.markdown("""
    **ที่มา:**  
    - ข้อมูลในโปรเจคนี้ถูกสร้างโดย **ChatGPT**

    **Dataset 1: Health Data (ข้อมูลสุขภาพ)**  
    - **ฟีเจอร์:**  
      - **ID:** รหัสประจำตัวผู้เข้าร่วม  
      - **BMI:** ดัชนีมวลกาย (numeric)  
      - **BloodPressure:** ความดันโลหิต (numeric)  
      - **HeartRate:** อัตราการเต้นของหัวใจ (numeric)  
      - **Risk:** ระดับความเสี่ยงด้านสุขภาพ (Categorical: "Low", "Medium", "High")  
    - **ความไม่สมบูรณ์:**  
      - ฟีเจอร์ BMI, BloodPressure และ HeartRate มี missing values ประมาณ 10%

    **Dataset 2: Financial Data (ข้อมูลการเงิน)**  
    - **ฟีเจอร์:**  
      - **ID:** รหัสประจำตัวข้อมูล  
      - **StockPrice:** ราคาหุ้น (numeric)  
      - **Income:** รายรับ (numeric)  
      - **Expense:** รายจ่าย (numeric)  
      - **NetProfit:** กำไรสุทธิ (Target สำหรับ Regression) คำนวณจาก Income - Expense พร้อม noise เล็กน้อย  
    - **ความไม่สมบูรณ์:**  
      - ฟีเจอร์ StockPrice, Income และ Expense มี missing values ประมาณ 10%
    """)

# ==========================================================
# Tab 2: Machine Learning (สำหรับ Health Data)
# ==========================================================
with tabs[1]:
    st.title("Machine Learning: Health Data")
    st.markdown("""
    **การเตรียมข้อมูล:**  
    - เติม missing values ด้วยค่าเฉลี่ยสำหรับ BMI, BloodPressure, และ HeartRate  
      
    **ทฤษฎีของอัลกอริทึม:**  
    - **Decision Tree:**  
      - ใช้จำแนกระดับความเสี่ยง (Risk) โดยแบ่งข้อมูลตามเงื่อนไขของฟีเจอร์  
      - สามารถแสดงภาพต้นไม้ตัดสินใจเพื่อเข้าใจการตัดสินใจ  
    - **K-Means Clustering:**  
      - ใช้จัดกลุ่มข้อมูลสุขภาพที่มีลักษณะคล้ายกันโดยไม่ใช้ target  
      
    **ขั้นตอนการพัฒนา:**  
    1. เตรียมข้อมูลด้วยการเติม missing values  
    2. แบ่งข้อมูลสำหรับฝึกโมเดล (Decision Tree)  
    3. ฝึกโมเดลและปรับแต่ง hyperparameters  
    4. ประเมินและวิเคราะห์ผลลัพธ์
    """)
    st.subheader("ตัวอย่างข้อมูล Health Data")
    df_health = load_data("data/health_dataset.csv")
    st.dataframe(df_health.head(10))

# ==========================================================
# Tab 3: Neural Network (สำหรับ Financial Data)
# ==========================================================
with tabs[2]:
    st.title("Neural Network: Financial Data")
    st.markdown("""
    **การเตรียมข้อมูล:**  
    - เติม missing values ด้วยค่าเฉลี่ยสำหรับ StockPrice, Income, และ Expense  
      
    **ทฤษฎีของ Neural Network:**  
    - ใช้โครงสร้าง Neural Network เพื่อจับความสัมพันธ์เชิงซับซ้อน  
    - สำหรับปัญหา Regression โดยมี target เป็น NetProfit  
      
    **ขั้นตอนการพัฒนา:**  
    1. เตรียมข้อมูลและแยกชุด train/test  
    2. สร้างโครงสร้าง Neural Network  
    3. ฝึกโมเดลด้วย optimizer (เช่น Adam) และ loss function (เช่น MSE)  
    4. ประเมินและปรับปรุงโมเดล
    """)
    st.subheader("ตัวอย่างข้อมูล Financial Data")
    df_financial = load_data("data/financial_dataset.csv")
    st.dataframe(df_financial.head(10))

# ==========================================================
# Tab 4: Demo Machine Learning (สำหรับ Health Data)
# ==========================================================
with tabs[3]:
    st.title("Demo Machine Learning: Health Data")
    
    st.markdown("### ส่วนที่ 1: Decision Tree Classification")
    with st.form("form_dt"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ปรับแต่งโมเดล Decision Tree**")
            dt_max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, step=1)
            dt_min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
        with col2:
            st.markdown("**ป้อนค่า Feature สำหรับทำนาย**")
            input_bmi = st.number_input("BMI", value=25.0, key="dt_bmi_input")
            input_bp = st.number_input("BloodPressure", value=120.0, key="dt_bp_input")
            input_hr = st.number_input("HeartRate", value=75.0, key="dt_hr_input")
        submitted_dt = st.form_submit_button("ทำนายด้วย Decision Tree")
    
    # เตรียมข้อมูล Health Data
    df_health_demo = load_data("data/health_dataset.csv")
    df_health_demo = prepare_health_dataset(df_health_demo)
    # เข้ารหัส target Risk
    le_risk = LabelEncoder()
    df_health_demo['Risk_enc'] = le_risk.fit_transform(df_health_demo['Risk'])
    # เตรียม features และ target สำหรับโมเดล
    X_health = df_health_demo[['BMI', 'BloodPressure', 'HeartRate']]
    y_health = df_health_demo['Risk_enc']
    X_train, X_test, y_train, y_test = train_test_split(X_health, y_health, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, random_state=42)
    dt_model.fit(X_train, y_train)
    
    if submitted_dt:
        input_features = np.array([[input_bmi, input_bp, input_hr]])
        prediction = dt_model.predict(input_features)
        predicted_risk = le_risk.inverse_transform(prediction)[0]
        st.success(f"ผลลัพธ์การจำแนกความเสี่ยง (Decision Tree): {predicted_risk}")
        
        # แสดงภาพต้นไม้ตัดสินใจ
        fig_tree, ax_tree = plt.subplots(figsize=(12,8))
        plot_tree(dt_model, filled=True, feature_names=['BMI', 'BloodPressure', 'HeartRate'], class_names=le_risk.classes_)
        st.pyplot(fig_tree)
        
        # แสดงความสำคัญของฟีเจอร์
        importances = dt_model.feature_importances_
        fig_imp, ax_imp = plt.subplots(figsize=(8,6))
        ax_imp.bar(['BMI', 'BloodPressure', 'HeartRate'], importances, color='skyblue')
        ax_imp.set_title("Feature Importances in Decision Tree")
        st.pyplot(fig_imp)
    
    st.markdown("---")
    st.markdown("### ส่วนที่ 2: K-Means Clustering")
    n_clusters = st.slider("จำนวนคลัสเตอร์", min_value=2, max_value=10, value=3, step=1, key="km_clusters")
    if st.button("รัน K-Means", key="btn_km"):
        X_km = df_health_demo[['BMI', 'BloodPressure', 'HeartRate']]
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans_model.fit_predict(X_km)
        df_health_demo['Cluster'] = clusters
        st.success(f"K-Means แบ่งออกเป็น {n_clusters} คลัสเตอร์แล้ว")
        fig_km, ax_km = plt.subplots(figsize=(8,6))
        sns.scatterplot(x='BMI', y='BloodPressure', hue='Cluster', data=df_health_demo, palette='viridis', ax=ax_km)
        ax_km.set_title("K-Means Clustering ของ Health Data")
        st.pyplot(fig_km)

# ==========================================================
# Tab 5: Demo Neural Network (สำหรับ Financial Data)
# ==========================================================
with tabs[4]:
    st.title("Demo Neural Network: Financial Data")
    st.markdown("ป้อนค่าฟีเจอร์เพื่อทำนาย NetProfit")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_stock = st.number_input("StockPrice", value=200.0, key="nn_stock")
    with col2:
        input_income = st.number_input("Income", value=75000.0, key="nn_income")
    with col3:
        input_expense = st.number_input("Expense", value=50000.0, key="nn_expense")
    if st.button("ทำนายด้วย Neural Network", key="btn_nn"):
        nn_model = load_nn_model()
        if nn_model is not None:
            input_data = np.array([[input_stock, input_income, input_expense]])
            pred_netprofit = nn_model.predict(input_data)
            st.success(f"NetProfit ที่ทำนายได้: {pred_netprofit[0][0]:.2f}")
        else:
            st.error("โมเดล Neural Network ไม่พร้อมใช้งาน กรุณาฝึกโมเดลก่อน")
    st.subheader("Distribution ของ NetProfit ใน Financial Data")
    df_financial_demo = load_data("data/financial_dataset.csv")
    fig_nn, ax_nn = plt.subplots(figsize=(8,6))
    sns.histplot(df_financial_demo['NetProfit'], kde=True, ax=ax_nn)
    ax_nn.set_title("Distribution of NetProfit")
    st.pyplot(fig_nn)
