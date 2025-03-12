import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ตั้งค่า layout ของหน้าเว็บ
st.set_page_config(page_title="IS_ML/NN Project", layout="wide")

# ----- ฟังก์ชันช่วยโหลดข้อมูล CSV -----
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

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

# ----- สร้างแท็บสำหรับ Navigation -----
tabs = st.tabs([
    "🏠 Home", 
    "📘 ML Explanation", 
    "📙 NN Explanation", 
    "🤖 ML Model Demo", 
    "🧠 NN Model Demo"
])

# ==========================================================
# Tab 1: Overview
# ==========================================================
with tabs[0]:
    st.title("🔍 Overview ของโปรเจค")
    st.markdown("""
    **1. ที่มาของ Dataset:**  
    - Dataset ทั้งหมดในโปรเจคนี้ถูกสร้างขึ้นโดย **ChatGPT**

    **2. Feature ของ Dataset:**  
    - **Financial Data:**  
      - **ID:** รหัสประจำตัวของข้อมูล  
      - **StockPrice:** ราคาหุ้น (numeric)  
      - **Income:** รายรับ (numeric)  
      - **Expense:** รายจ่าย (numeric)  
      - **NetProfit:** กำไรสุทธิ (คำนวณจาก Income - Expense พร้อม noise เล็กน้อย)
      
    - **Synthetic Digit Images:**  
      - รูปภาพขนาด 64x64 pixel ที่มีตัวเลขจาก MNIST กระจายอยู่ในภาพ  
      - **Label:** เวกเตอร์ขนาด 10 ค่า ระบุจำนวนของตัวเลขแต่ละตัว (0-9)

    **3. ความไม่สมบูรณ์ของ Dataset:**  
    - **Financial Data:**  
      - มี missing values ใน StockPrice, Income, และ Expense (ประมาณ 10%)  
    - **Synthetic Digit Images:**  
      - ตัวเลขในภาพถูกวางแบบสุ่มในตำแหน่งและขนาดที่แตกต่างกัน  
      - บางครั้ง label อาจจะหายหรือไม่ครบถ้วน
    """)

# ==========================================================
# Tab 2: Machine Learning Explanation (สำหรับ Financial Data)
# ==========================================================
with tabs[1]:
    st.title("📘 Machine Learning Explanation")
    
    st.subheader("Financial Data (Decision Tree & K-Means)")
    st.markdown("""
    **แนวทางการพัฒนา:**  
    1. **การเตรียมข้อมูล:**  
       - โหลดข้อมูลและเติม missing values ด้วยค่าเฉลี่ยสำหรับ StockPrice, Income, และ Expense
       - คำนวณ NetProfit (Income - Expense + noise) และสร้าง target ใหม่ **Profit_Class**  
         โดยกำหนดให้ Profit_Class = 1 หาก NetProfit อยู่เหนือค่า median, และ 0 หากต่ำกว่า

    2. **ทฤษฎีอัลกอริทึม:**  
       - **Decision Tree:**  
         เป็นอัลกอริทึมการจำแนกประเภท (Classification) ที่แบ่งข้อมูลตาม feature เพื่อลดความสับสนในการตัดสินใจ  
       - **K-Means Clustering:**  
         เป็นอัลกอริทึมการจัดกลุ่ม (Unsupervised Learning) ที่แบ่งข้อมูลออกเป็นคลัสเตอร์ตามความใกล้เคียงของ feature

    3. **ขั้นตอนการพัฒนาโมเดล:**  
       - แบ่งข้อมูลเป็น training/test set  
       - ฝึก Decision Tree เพื่อจำแนก Profit_Class  
       - ฝึก K-Means เพื่อจัดกลุ่มข้อมูลทางการเงิน  
       - ประเมินผลและแสดงผลลัพธ์ เช่น ความแม่นยำ และการวิเคราะห์กลุ่ม (cluster analysis)

    """)

    st.subheader("ตัวอย่าง Dataset Financial Data ก่อนเตรียมข้อมูล:")
    df_financial = load_data("data/financial_dataset.csv")
    st.dataframe(df_financial.head(10))

    st.subheader("ตัวอย่าง Dataset Financial Data หลังเตรียมข้อมูล:")
    df_financial = load_data("data/financial_dataset_prepared.csv")
    st.dataframe(df_financial.head(10))

# ==========================================================
# Tab 3: Neural Network Explanation (สำหรับ Synthetic Digit Images)
# ==========================================================
with tabs[2]:
    st.title("📙 Neural Network Explanation")
    
    st.subheader("Synthetic Digit Images (CNN)")
    st.markdown("""
    **แนวทางการพัฒนา:**  
    1. **การเตรียมข้อมูล:**  
       - สร้าง synthetic image โดยสุ่มเลือกตัวเลขจาก MNIST แล้ววางลงบน canvas ขนาด 64x64 pixel  
       - สร้าง label vector ขนาด 10 ค่า ที่ระบุจำนวนของตัวเลขแต่ละตัวในภาพ  
       - จัดการกับความไม่สมบูรณ์ของ label (เช่น กรณีที่ label หาย) โดยการแทนที่ด้วยเวกเตอร์ [0, 0, ..., 0] ซึ่งหมายความว่าไม่มีตัวเลขปรากฏในภาพ

    2. **ทฤษฎีอัลกอริทึม (CNN):**  
       - ใช้ Convolutional Neural Network (CNN) เพื่อเรียนรู้คุณลักษณะของภาพ  
       - สถาปัตยกรรมประกอบด้วย Convolutional Layers, MaxPooling Layers, Flatten, Dense Layers และ Dropout

    3. **ขั้นตอนการพัฒนาโมเดล:**  
       - โหลดและเตรียมข้อมูลภาพและ label  
       - ออกแบบสถาปัตยกรรม CNN และกำหนด hyperparameters  
       - ฝึกโมเดลและประเมินผลด้วย test set
       - การใช้โมเดล CNN สำหรับการนับจำนวนตัวเลขในภาพ
       - แสดงผลลัพธ์ว่ามีเลขและจำนวนของแต่ละเลขอะไรบ้าง โดยใช้ค่าพยากรณ์ที่มากกว่า 0.5 เป็นเกณฑ์ในการตัดสินใจ
    """)

    st.subheader("ตัวอย่าง Dataset Synthetic Digit Images ก่อนเตรียมข้อมูล:")
    df_labels = load_data("data/digits/labels.csv")
    st.dataframe(df_labels.head(10))
    num_examples = 5
    for index, row in df_labels.head(num_examples).iterrows():
        st.markdown(f"**Filename:** {row['filename']} | **Label:** {row['label']}")
        image_path = os.path.join("data/digits/images", row["filename"])
        st.image(image_path, caption=row["filename"], width=150)


    st.subheader("ตัวอย่าง Dataset Synthetic Digit Images หลังเตรียมข้อมูล:")
    df_labels = load_data("data/digits/labels_prepared.csv")
    st.dataframe(df_labels.head(10))
    num_examples = 5
    for index, row in df_labels.head(num_examples).iterrows():
        st.markdown(f"**Filename:** {row['filename']} | **Label:** {row['label']}")
        image_path = os.path.join("data/digits/images", row["filename"])
        st.image(image_path, caption=row["filename"], width=150)

# ==========================================================
# Tab 4: Demo Machine Learning Model (Financial Data)
# ==========================================================
with tabs[3]:
    st.title("🤖 Demo ML Model (Financial Data)")
    
    st.markdown("## ส่วนที่ 1: Decision Tree Classification")
    st.markdown("""
    ในส่วนนี้จะใช้ข้อมูล Financial Data ที่มี feature **StockPrice**, **Income** และ **Expense**  
    โดยคำนวณ **NetProfit** จากข้อมูลที่มีอยู่และสร้าง target ใหม่ **Profit_Class**  
    (Profit_Class = 1 หาก NetProfit อยู่เหนือค่า median, 0 หากต่ำกว่า)  
    จากนั้นฝึกโมเดล Decision Tree เพื่อจำแนกประเภทของ Profit_Class  
    """)
    
    with st.form("form_dt"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**ปรับแต่งโมเดล Decision Tree**")
            dt_max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, step=1)
            dt_min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
        with col2:
            st.markdown("**ป้อนค่า Feature**")
            input_stock = st.number_input("StockPrice", value=200.0, key="dt_stock_input")
        with col3:
            st.markdown("**ป้อนค่า Feature**")
            input_income = st.number_input("Income", value=75000.0, key="dt_income_input")
            input_expense = st.number_input("Expense", value=50000.0, key="dt_expense_input")
        submitted_dt = st.form_submit_button("ทำนายด้วย Decision Tree")
    
    # โหลดและเตรียมข้อมูล Financial Data
    df_financial_demo = load_data("data/financial_dataset_prepared.csv")
    df_financial_demo = prepare_financial_dataset(df_financial_demo)
    # คำนวณ Profit_Class โดยใช้ median ของ NetProfit
    median_profit = df_financial_demo['NetProfit'].median()
    df_financial_demo['Profit_Class'] = (df_financial_demo['NetProfit'] >= median_profit).astype(int)
    
    # เตรียม features และ target สำหรับโมเดล
    X_financial = df_financial_demo[['StockPrice', 'Income', 'Expense']]
    y_financial = df_financial_demo['Profit_Class']
    X_train, X_test, y_train, y_test = train_test_split(X_financial, y_financial, test_size=0.2, random_state=42)
    
    dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, random_state=42)
    dt_model.fit(X_train, y_train)
    
    if submitted_dt:
        input_features = np.array([[input_stock, input_income, input_expense]])
        prediction = dt_model.predict(input_features)[0]
        result = "High Profit" if prediction == 1 else "Low Profit"
        st.success(f"ผลลัพธ์การจำแนก Profit_Class: {result}")
        
        # แสดงภาพต้นไม้ของ Decision Tree
        fig_tree, ax_tree = plt.subplots(figsize=(12,8))
        plot_tree(dt_model, filled=True, feature_names=['StockPrice', 'Income', 'Expense'], class_names=["Low Profit", "High Profit"])
        st.pyplot(fig_tree)
        
        # แสดงความสำคัญของฟีเจอร์
        importances = dt_model.feature_importances_
        fig_imp, ax_imp = plt.subplots(figsize=(8,6))
        ax_imp.bar(['StockPrice', 'Income', 'Expense'], importances, color='skyblue')
        ax_imp.set_title("Feature Importances in Decision Tree")
        st.pyplot(fig_imp)
    
    st.markdown("---")
    st.markdown("## ส่วนที่ 2: K-Means Clustering สำหรับ Financial Data")
    n_clusters = st.slider("จำนวนคลัสเตอร์", min_value=2, max_value=10, value=3, step=1, key="km_clusters")
    if st.button("รัน K-Means", key="btn_km"):
        X_financial_all = df_financial_demo[['StockPrice', 'Income', 'Expense']]
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans_model.fit_predict(X_financial_all)
        df_financial_demo['Cluster'] = clusters
        st.success(f"K-Means แบ่งออกเป็น {n_clusters} คลัสเตอร์แล้ว")
        # Visualization โดยใช้ Income กับ Expense เป็นตัวอย่าง
        fig_km, ax_km = plt.subplots(figsize=(8,6))
        sns.scatterplot(x='Income', y='Expense', hue='Cluster', data=df_financial_demo, palette='viridis', ax=ax_km)
        ax_km.set_title("K-Means Clustering (Income vs Expense)")
        st.pyplot(fig_km)

# ==========================================================
# Tab 5: Demo Neural Network Model (Synthetic Digit Images)
# ==========================================================
with tabs[4]:
    st.title("🧠 Demo NN Model (Synthetic Digit Images)")
    
    # ให้ผู้ใช้เลือกภาพตัวอย่างจาก Synthetic Digit Images
    df_labels = load_data("data/digits/labels_prepared.csv")
    image_options = df_labels["filename"].tolist()
    selected_image = st.selectbox("เลือกตัวอย่างภาพ", image_options)
    
    # ดึง label ของภาพที่เลือกมาแสดง
    selected_label = df_labels[df_labels["filename"] == selected_image]["label"].values[0]
    st.markdown(f"**Label ของภาพ:** {selected_label}")
    
    image_path = os.path.join("data/digits/images", selected_image)
    img = load_img(image_path, color_mode="grayscale", target_size=(64, 64))
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
            
            # แสดงผลลัพธ์ว่ามีเลขอะไรบ้างในรูปและจำนวนของแต่ละเลข
            predicted_digits = {str(i): int(round(count)) for i, count in enumerate(prediction) if count > 0.5}
            st.markdown("ตัวเลขที่พบในภาพและจำนวนของแต่ละเลข:")
            for digit, count in predicted_digits.items():
                st.markdown(f"ตัวเลข {digit}: {count} ตัว")
            
            # แสดงกราฟแท่งสำหรับผลการพยากรณ์
            fig, ax = plt.subplots(figsize=(10,6))
            ax.bar(range(10), prediction, color='skyblue')
            ax.set_xlabel("Number (0-9)")
            ax.set_ylabel("Prediction value")
            ax.set_title("Predicting the number of digits in an image")
            st.pyplot(fig)


