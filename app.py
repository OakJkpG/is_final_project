# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ตั้งค่า layout ของหน้าเว็บ
st.set_page_config(page_title="ML&NN IS_Project", layout="wide")

# ----- Custom CSS สำหรับตกแต่งหน้าเว็บ -----
st.markdown(
    """
    <style>
    /* เปลี่ยนสีพื้นหลังและ font */
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* ตกแต่งปุ่ม */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
    }
    /* ตกแต่งแท็บ */
    .css-1outpf7 {
        background-color: #ffffff;
        border-bottom: 2px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----- ฟังก์ชันช่วยโหลดข้อมูล CSV -----
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# ----- ฟังก์ชันสำหรับการเตรียมข้อมูล Dataset1 (สำหรับ Demo ML) -----
def prepare_dataset1(df):
    # เติม missing values สำหรับตัวเลข
    df['Feature_A'].fillna(df['Feature_A'].mean(), inplace=True)
    df['Feature_B'].fillna(df['Feature_B'].mean(), inplace=True)
    # เติม missing values สำหรับตัวแปร categorical ด้วย mode
    df['Feature_C'].fillna(df['Feature_C'].mode()[0], inplace=True)
    # เข้ารหัส Feature_C
    le = LabelEncoder()
    df['Feature_C_enc'] = le.fit_transform(df['Feature_C'])
    return df, le

# ----- ฟังก์ชันสำหรับโหลดโมเดล Neural Network ที่ฝึกไว้ -----
@st.cache_resource
def load_nn_model():
    model = load_model('models/neural_network.h5')
    return model

# ----- ฟังก์ชันสำหรับโหลด pre-saved label encoders (ถ้ามี) -----
@st.cache_resource
def load_label_encoders():
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return encoders

# ----- สร้าง Navigation แบบแท็บด้านบน -----
tabs = st.tabs(["🏠 Home", "📘 ML Explanation", "📙 NN Explanation", "🤖 ML Model Demo", "🧠 NN Model Demo"])

# ==========================================================
# Tab 1: Overview
# ==========================================================
with tabs[0]:
    st.title("Overview ของโปรเจค")
    st.write("""
        **รายละเอียดโปรเจค:**  
        โปรเจคนี้เป็นตัวอย่างการพัฒนาโมเดล Machine Learning และ Neural Network โดยใช้ข้อมูลที่มีความไม่สมบูรณ์  
        โดยแบ่งออกเป็น 2 ชุดข้อมูลหลัก:
        - **Dataset1:** สำหรับพัฒนาโมเดล Machine Learning เช่น การจำแนกประเภทด้วย **Decision Tree** และการจัดกลุ่มด้วย **K-Means**
        - **Dataset2:** สำหรับพัฒนาโมเดล **Neural Network** ในปัญหา Regression  
        
        **เป้าหมาย:**  
        - ทำความเข้าใจและเตรียมข้อมูลที่มีความไม่สมบูรณ์  
        - อธิบายทฤษฎีและแนวทางของอัลกอริทึมที่ใช้  
        - สาธิตการทำงานของโมเดลแต่ละประเภทผ่านเว็บแอปพลิเคชันที่พัฒนาโดยใช้ Streamlit  
    """)
    #st.image("https://via.placeholder.com/1000x300.png?text=Machine+Learning+%26+Neural+Network+Project", use_column_width=True)

# ==========================================================
# Tab 2: Machine Learning (อธิบายการพัฒนา ML)
# ==========================================================
with tabs[1]:
    st.title("Machine Learning: Data Preparation & Model Development")
    st.write("""
        **การเตรียมข้อมูล:**  
        - **Dataset1:** ข้อมูลที่มี missing values ในตัวเลขและตัวแปรเชิงประเภท  
        - เติม missing values ด้วยค่าเฉลี่ย (สำหรับตัวเลข) และ mode (สำหรับ categorical)  
        - ใช้ LabelEncoder เพื่อแปลงข้อมูลเชิงประเภทให้เป็นตัวเลข  
        
        **ทฤษฎีของอัลกอริทึม:**  
        - **Decision Tree:**  
            - แบ่งข้อมูลออกเป็นสาขาตามเงื่อนไขของแต่ละ feature  
            - ง่ายต่อการตีความและเข้าใจการตัดสินใจ  
        - **K-Means Clustering:**  
            - จัดกลุ่มข้อมูลออกเป็นคลัสเตอร์ตามความคล้ายคลึงกัน  
            - ไม่ต้องใช้ target ในการฝึก (unsupervised learning)  
        
        **ขั้นตอนการพัฒนา:**  
        1. เตรียมข้อมูลให้สมบูรณ์  
        2. แบ่งข้อมูล (สำหรับ supervised learning)  
        3. ฝึกโมเดลและปรับแต่ง hyperparameters  
        4. ประเมินผลและวิเคราะห์การทำงานของโมเดล  
    """)
    st.subheader("ตัวอย่างข้อมูล Dataset1")
    df1_sample = load_data('data/dataset1.csv').head(10)
    st.dataframe(df1_sample)

# ==========================================================
# Tab 3: Neural Network (อธิบายการพัฒนา NN)
# ==========================================================
with tabs[2]:
    st.title("Neural Network: Data Preparation & Model Development")
    st.write("""
        **การเตรียมข้อมูล:**  
        - **Dataset2:** ข้อมูลจากเซนเซอร์ที่มี missing values  
        - เติม missing values ด้วยค่าเฉลี่ยในแต่ละ Sensor เพื่อให้ข้อมูลสมบูรณ์
        
        **ทฤษฎีของ Neural Network:**  
        - ได้รับแรงบันดาลใจจากโครงสร้างของสมอง  
        - ใช้เลเยอร์ Dense (fully-connected) เพื่อจับความสัมพันธ์ในข้อมูล  
        - ใช้ Dropout เพื่อลดปัญหา overfitting  
        - สำหรับปัญหา Regression จะมี Output layer เดียว  
        
        **ขั้นตอนการพัฒนา:**  
        1. เตรียมข้อมูล: ทำความสะอาดและแยกชุดข้อมูล train/test  
        2. สร้างโครงสร้างโมเดล: กำหนดจำนวนเลเยอร์และจำนวนหน่วยในแต่ละเลเยอร์  
        3. ฝึกโมเดลด้วย optimizer (เช่น Adam) และ loss function (เช่น MSE)  
        4. ประเมินและปรับปรุงโมเดล  
    """)
    st.subheader("ตัวอย่างข้อมูล Dataset2")
    df2_sample = load_data('data/dataset2.csv').head(10)
    st.dataframe(df2_sample)

# ==========================================================
# Tab 4: Demo Machine Learning (Decision Tree & K-Means รวมกัน)
# ==========================================================
with tabs[3]:
    st.title("Demo Machine Learning")
    
    st.markdown("## Decision Tree Classification")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ปรับแต่งโมเดล Decision Tree")
        dt_max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, step=1)
        dt_min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
    
    with col2:
        st.markdown("### ป้อนค่า Feature สำหรับทำนาย")
        feature_A = st.number_input("Feature_A", value=50.0, key="dt_feature_A")
        feature_B = st.number_input("Feature_B", value=50.0, key="dt_feature_B")
        feature_C = st.selectbox("Feature_C", options=["Low", "Medium", "High"], key="dt_feature_C")
    
    # โหลดและเตรียมข้อมูล Dataset1
    df1 = load_data('data/dataset1.csv')
    df1, le = prepare_dataset1(df1)
    # สร้าง target encoding สำหรับ Decision Tree
    le_target = LabelEncoder()
    df1['Target_enc'] = le_target.fit_transform(df1['Target'])
    
    # เตรียมข้อมูลสำหรับการฝึก Decision Tree
    X = df1[['Feature_A', 'Feature_B', 'Feature_C_enc']]
    y = df1['Target_enc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ฝึก Decision Tree ด้วย hyperparameters ที่เลือก
    dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, random_state=42)
    dt_model.fit(X_train, y_train)
    
    if st.button("ทำนายด้วย Decision Tree", key="btn_dt"):
        feature_C_enc = le.transform([feature_C])[0]
        input_features = np.array([[feature_A, feature_B, feature_C_enc]])
        prediction = dt_model.predict(input_features)
        predicted_class = le_target.inverse_transform(prediction)[0]
        st.success(f"ผลลัพธ์การจำแนกประเภท (Decision Tree): {predicted_class}")
    
    st.markdown("---")
    # decision tree
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # สมมติว่า dt_model คือโมเดล Decision Tree ที่ฝึกไว้ และ le_target คือ LabelEncoder สำหรับ target
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(dt_model, filled=True, feature_names=['Feature_A', 'Feature_B', 'Feature_C_enc'], class_names=le_target.classes_)
    st.pyplot(fig)

    st.markdown("## K-Means Clustering")
    n_clusters = st.slider("จำนวนคลัสเตอร์ (Clusters)", min_value=2, max_value=10, value=3, step=1, key="km_clusters")
    if st.button("รัน K-Means", key="btn_km"):
        X_km = df1[['Feature_A', 'Feature_B', 'Feature_C_enc']]
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans_model.fit_predict(X_km)
        df1['Cluster'] = clusters
        
        st.success(f"K-Means แบ่งออกเป็น {n_clusters} คลัสเตอร์แล้ว")
        
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x='Feature_A', y='Feature_B', hue='Cluster', data=df1, palette='viridis', ax=ax)
        ax.set_title("K-Means Clustering ของ Dataset1")
        st.pyplot(fig)

# ==========================================================
# Tab 5: Demo Neural Network
# ==========================================================
with tabs[4]:
    st.title("Demo Neural Network for Regression")
    st.write("ป้อนค่าจากเซนเซอร์เพื่อทำนายค่า Output ด้วย Neural Network")
    
    sensor1 = st.number_input("Sensor1", value=20.0, key="nn_sensor1")
    sensor2 = st.number_input("Sensor2", value=50.0, key="nn_sensor2")
    sensor3 = st.number_input("Sensor3", value=100.0, key="nn_sensor3")
    
    nn_model = load_nn_model()
    
    if st.button("ทำนายด้วย Neural Network"):
        input_data = np.array([[sensor1, sensor2, sensor3]])
        pred_output = nn_model.predict(input_data)
        st.success(f"ผลลัพธ์การทำนาย Output: {pred_output[0][0]:.2f}")
    
    st.subheader("Distribution ของ Output ใน Dataset2")
    df2 = load_data('data/dataset2.csv')
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df2['Output'], kde=True, ax=ax)
    ax.set_title("Distribution of Output in Dataset2")
    st.pyplot(fig)
