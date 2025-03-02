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
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ตั้งค่า layout ของหน้าเว็บ
st.set_page_config(page_title="ML/NN Project", layout="wide")

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

# ----- ฟังก์ชันสำหรับเตรียมข้อมูล Health Data -----
def prepare_health_dataset(df):
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['HeartRate'] = df['HeartRate'].fillna(df['HeartRate'].mean())
    return df

# ----- ฟังก์ชันสำหรับเตรียมข้อมูล Financial Data -----
def prepare_financial_dataset(df):
    df['StockPrice'] = df['StockPrice'].fillna(df['StockPrice'].mean())
    df['Income'] = df['Income'].fillna(df['Income'].mean())
    df['Expense'] = df['Expense'].fillna(df['Expense'].mean())
    return df

# ----- ฟังก์ชันสำหรับโหลดโมเดล CNN (สำหรับ Synthetic Digit Images) -----
@st.cache_resource
def load_cnn_model():
    model_path = "models/digit_count_cnn.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

# ----- ฟังก์ชันช่วยโหลด label สำหรับ Synthetic Digit Images -----
@st.cache_data
def load_labels():
    labels_path = os.path.join("data/digits", "labels.csv")
    return pd.read_csv(labels_path)

# ----- สร้างแท็บสำหรับ Navigation -----
tabs = st.tabs([
    "🏠 Home", 
    "📘 ML Explanation", 
    "📙 NN Explanation", 
    "🤖 ML Model Demo", 
    "🧠 Demo CNN"
])

# ==========================================================
# Tab 1: Overview
# ==========================================================
with tabs[0]:
    st.title("🔍 Overview ของโปรเจค")
    st.markdown("""
    **ที่มา:**  
    - ข้อมูลในโปรเจคนี้ถูกสร้างโดย **ChatGPT**

    **Dataset 1: Health Data (ข้อมูลสุขภาพ)**  
    - **Features:**  
      - **ID:** รหัสประจำตัวผู้เข้าร่วม  
      - **BMI:** ดัชนีมวลกาย (numeric)  
      - **BloodPressure:** ความดันโลหิต (numeric)  
      - **HeartRate:** อัตราการเต้นของหัวใจ (numeric)  
      - **Risk:** ระดับความเสี่ยงด้านสุขภาพ (Categorical: "Low", "Medium", "High")  
    - **ความไม่สมบูรณ์:**  
      - ฟีเจอร์ BMI, BloodPressure และ HeartRate มี missing values ประมาณ 10%
    - **ML Approach:** 
      - ใช้ Decision Tree Classification เพื่อจำแนกระดับความเสี่ยง

    **Dataset 2: Financial Data (ข้อมูลการเงิน)**  
    - **Features:**  
      - **ID:** รหัสประจำตัวข้อมูล  
      - **StockPrice:** ราคาหุ้น (numeric)  
      - **Income:** รายรับ (numeric)  
      - **Expense:** รายจ่าย (numeric)  
      - **NetProfit:** กำไรสุทธิ (Target สำหรับ Regression) คำนวณจาก Income - Expense พร้อม noise เล็กน้อย  
    - **ความไม่สมบูรณ์:**  
      - ฟีเจอร์ StockPrice, Income และ Expense มี missing values ประมาณ 10%
    - **ML Approach:** 
      - ใช้ K-Means Clustering เพื่อจัดกลุ่มข้อมูลทางการเงิน
      
    **Dataset 3: Synthetic Digit Images (ข้อมูลภาพตัวเลขสังเคราะห์)**  
    - **Features:**  
      - รูปภาพขนาด 64x64 พิกเซล ที่มีตัวเลขจาก MNIST กระจายอยู่ในภาพแบบสุ่ม  
      - แต่ละภาพมีตัวเลข 3-7 ตัว 
      - สร้าง label vector ขนาด 10 ค่า ระบุจำนวนของตัวเลข 0-9    
    - **ความไม่สมบูรณ์:**  
      - ตัวเลขในแต่ละภาพถูกวางในตำแหน่งและขนาดที่แปรปรวน เพื่อจำลองข้อมูลจริง
    - **DL Approach:** 
      - ใช้ Convolutional Neural Network (CNN) ในการนับจำนวนตัวเลข
    """)
    
# ==========================================================
# Tab 2: Machine Learning Explanation (สำหรับ Health & Financial Data)
# ==========================================================
with tabs[1]:
    st.title("📘 Machine Learning Explanation")
    st.subheader("1. Health Data (Decision Tree Classification)")
    st.markdown("""
    **การเตรียมข้อมูล:**  
    - เติม missing values ด้วยค่าเฉลี่ยสำหรับ BMI, BloodPressure, และ HeartRate

    **อัลกอริทึม Decision Tree:**  
    - ใช้สำหรับจำแนกประเภทระดับความเสี่ยง (Risk)  
    - สามารถแสดงภาพต้นไม้ตัดสินใจเพื่อวิเคราะห์การตัดสินใจ   

    **ขั้นตอนการพัฒนา:**  
    1. เตรียมข้อมูลด้วยการเติม missing values  
    2. แบ่งข้อมูลสำหรับฝึกโมเดล (Train/Test Split)  
    3. ฝึกโมเดล Decision Tree และปรับแต่ง hyperparameters  
    4. ประเมินและวิเคราะห์ผลลัพธ์
    """)
    st.subheader("ตัวอย่างข้อมูล Health Data")
    df_health = load_data("data/health_dataset.csv")
    st.dataframe(df_health.head(10))

    st.subheader("2. Financial Data (K-Means Clustering)")
    st.markdown("""
    **การเตรียมข้อมูล:**  
    - เติม missing values ด้วยค่าเฉลี่ยสำหรับ StockPrice, Income, และ Expense

    **อัลกอริทึม K-Means Clustering:**  
    - เป็นการเรียนรู้แบบไม่มีผู้สอน (Unsupervised Learning)  
    - ใช้สำหรับจัดกลุ่มข้อมูลทางการเงินตามลักษณะของข้อมูล

    **ขั้นตอนการพัฒนา:**  
    1. เตรียมข้อมูลด้วยการเติม missing values  
    2. แบ่งข้อมูลสำหรับฝึกโมเดล (K-Means)  
    3. ปรับแต่ง hyperparameters (เช่น จำนวนคลัสเตอร์)  
    4. วิเคราะห์และตรวจสอบผลการจัดกลุ่ม
    """)
    st.subheader("ตัวอย่างข้อมูล Financial Data")
    df_financial = load_data("data/financial_dataset.csv")
    st.dataframe(df_financial.head(10))

# ==========================================================
# Tab 3: Neural Network Explanation (สำหรับ Financial Data)
# ==========================================================
with tabs[2]:
    st.title("📙 Neural Network Explanation: Synthetic Digit Images")
    st.markdown("""
    **Synthetic Digit Images:**  
    - รูปภาพขนาด 64x64 พิกเซล ที่มีตัวเลขจาก MNIST ถูกวางแบบสุ่ม (3-7 ตัวต่อภาพ)  
    - **Label:** เวกเตอร์ 10 ค่า ระบุจำนวนของตัวเลขแต่ละตัว (0-9)
    
    **การเตรียมข้อมูล:**
    - สร้าง synthetic image โดยสุ่มเลือกตัวเลขจาก MNIST  
    - วางตัวเลขลงบน canvas ขนาด 64x64 พิกเซล โดยสุ่มตำแหน่งและจำนวน (3-7 ตัว)  
    - สร้าง label vector ขนาด 10 ค่า เพื่อระบุจำนวนของตัวเลขแต่ละตัว
    
    **อัลกอริทึม Convolutional Neural Network (CNN):**
    - ใช้สถาปัตยกรรม CNN ประกอบด้วย Convolutional Layers, MaxPooling Layers, และ Dense Layers  
    - โมเดลออกแบบให้สามารถทำนาย label vector สำหรับการนับจำนวนตัวเลขในภาพ

    **ขั้นตอนการพัฒนา:**
    1. โหลดและเตรียมข้อมูลภาพและ label จาก synthetic dataset  
    2. ออกแบบสถาปัตยกรรม CNN และกำหนด hyperparameters  
    3. ฝึกโมเดลโดยใช้ training set และปรับแต่ง model ให้มีประสิทธิภาพ  
    4. ประเมินผลโมเดลด้วย test set และตรวจสอบความถูกต้องในการนับตัวเลข
    """)
    st.subheader("ตัวอย่างข้อมูล Label (CSV) และภาพพร้อมข้อมูล Label")
    df_labels = load_labels()
    st.dataframe(df_labels.head(10))

    # แสดงตัวอย่าง 5 รูปพร้อมข้อมูล label
    num_examples = 5
    for index, row in df_labels.head(num_examples).iterrows():
        st.markdown(f"**Filename:** {row['filename']} | **Label:** {row['label']}")
        image_path = os.path.join("data/digits/images", row["filename"])
        st.image(image_path, caption=row["filename"], width=150)
        
# ==========================================================
# Tab 4: Demo Machine Learning Model
# ==========================================================
with tabs[3]:
    st.title("🤖 Demo ML Model")
    st.markdown("## ส่วนที่ 1: Decision Tree Classification (Health Data)")
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
    st.markdown("## ส่วนที่ 2: K-Means Clustering (Financial Data)")
    n_clusters = st.slider("จำนวนคลัสเตอร์", min_value=2, max_value=10, value=3, step=1, key="km_clusters")
    if st.button("รัน K-Means", key="btn_km"):
        df_financial_demo = load_data("data/financial_dataset.csv")
        df_financial_demo = prepare_financial_dataset(df_financial_demo)
        X_financial = df_financial_demo[['StockPrice', 'Income', 'Expense']]
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans_model.fit_predict(X_financial)
        df_financial_demo['Cluster'] = clusters
        st.success(f"K-Means แบ่งออกเป็น {n_clusters} คลัสเตอร์แล้ว")
        # สำหรับ visualization ใช้ Income กับ Expense เป็นตัวอย่าง
        fig_km, ax_km = plt.subplots(figsize=(8,6))
        sns.scatterplot(x='Income', y='Expense', hue='Cluster', data=df_financial_demo, palette='viridis', ax=ax_km)
        ax_km.set_title("K-Means Clustering of Financial Data (Income vs Expense)")
        st.pyplot(fig_km)

# ==========================================================
# Tab 5: Demo NN(CNN) Model for Digit Counting (Synthetic Digit Images)
# ==========================================================
with tabs[4]:
    st.title("🧠 Demo NN Model")
    
    # ให้ผู้ใช้เลือกภาพตัวอย่างจาก Synthetic Digit Images
    df_labels = load_labels()
    image_options = df_labels["filename"].tolist()
    selected_image = st.selectbox("เลือกตัวอย่างภาพ", image_options)
    
    image_path = os.path.join("data/digits/images", selected_image)
    img = load_img(image_path, color_mode="grayscale", target_size=(64,64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    st.image(image_path, caption=f"ตัวอย่าง: {selected_image}", use_container_width=False)
    
    # โหลดโมเดล CNN สำหรับ synthetic digit counting
    cnn_model = load_cnn_model()
    if cnn_model is None:
        st.error("โมเดล CNN ไม่พร้อมใช้งาน กรุณาฝึกโมเดลก่อน")
    else:
        if st.button("พยากรณ์จำนวนตัวเลขในภาพ", key="btn_cnn"):
            prediction = cnn_model.predict(img_array)[0]
            st.success(f"ผลการพยากรณ์: {np.round(prediction, 2)}")
            
            # แสดงกราฟแท่งสำหรับผลการพยากรณ์
            fig, ax = plt.subplots(figsize=(10,6))
            ax.bar(range(10), prediction, color='skyblue')
            ax.set_xlabel("Number (0-9)")
            ax.set_ylabel("Prediction value")
            ax.set_title("Predicting the number of digits in an image")
            st.pyplot(fig)
