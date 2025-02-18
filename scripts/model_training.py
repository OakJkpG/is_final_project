# model_training.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# นำเข้า module สำหรับเตรียมข้อมูล
from data_preparation import prepare_health_data, prepare_financial_data

##############################
# Training Models for Health Data (Machine Learning)
##############################
def train_health_models():
    # เตรียมข้อมูล Health Data
    df_health, le_risk = prepare_health_data()
    
    # ใช้ฟีเจอร์ BMI, BloodPressure, HeartRate เพื่อจำแนกประเภท Risk
    X = df_health[['BMI', 'BloodPressure', 'HeartRate']]
    y = df_health['Risk_enc']
    
    # แบ่งข้อมูลเป็น training และ testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ฝึก Decision Tree Classifier
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # ประเมินโมเดลด้วยข้อมูล test
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy on Health Data: {accuracy:.2f}")
    
    # ฝึก K-Means Clustering (unsupervised)
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(X)
    
    # สร้างโฟลเดอร์ models หากยังไม่มี
    os.makedirs("models", exist_ok=True)
    
    # บันทึกโมเดลและ LabelEncoder
    with open("models/health_decision_tree.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    with open("models/health_kmeans.pkl", "wb") as f:
        pickle.dump(kmeans_model, f)
    with open("models/health_label_encoder.pkl", "wb") as f:
        pickle.dump(le_risk, f)
    
    print("Health models saved in 'models/' directory.")

##############################
# Training Model for Financial Data (Neural Network)
##############################
def train_financial_nn():
    # เตรียมข้อมูล Financial Data
    df_financial = prepare_financial_data()
    
    # ใช้ฟีเจอร์ StockPrice, Income, Expense เพื่อทำนาย NetProfit
    X = df_financial[['StockPrice', 'Income', 'Expense']]
    y = df_financial['NetProfit']
    
    # แบ่งข้อมูลเป็น training และ testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # สร้างโมเดล Neural Network สำหรับ Regression
    model = Sequential([
        Dense(64, input_dim=3, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output สำหรับ Regression
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # ฝึกโมเดล
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)
    
    # ประเมินโมเดลด้วยชุด test
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network Test Loss on Financial Data: {loss:.2f}")
    
    # บันทึกโมเดล Neural Network
    os.makedirs("models", exist_ok=True)
    model.save("models/financial_nn.h5")
    print("Financial Neural Network model saved as 'models/financial_nn.h5'.")

if __name__ == "__main__":
    train_health_models()
    train_financial_nn()
