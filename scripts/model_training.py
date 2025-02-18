# scripts/model_training.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# นำเข้าโมดูลสำหรับเตรียมข้อมูล
from data_preparation import prepare_dataset1, prepare_dataset2

def train_dataset1_models():
    # เตรียมข้อมูล Dataset1
    df1, le_feature, le_target = prepare_dataset1(file_path='../IS_final_project/data/dataset1.csv')
    
    # บันทึก encoders เพื่อใช้ในเว็บแอป
    os.makedirs('../IS_final_project/models', exist_ok=True)
    with open('../IS_final_project/models/label_encoders.pkl', 'wb') as f:
        pickle.dump({'le_feature': le_feature, 'le_target': le_target}, f)
    
    # เตรียมข้อมูลสำหรับ Decision Tree
    X = df1[['Feature_A', 'Feature_B', 'Feature_C_enc']]
    y = df1['Target_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ฝึก Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    y_pred = dt_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", acc)
    
    # บันทึกโมเดล Decision Tree
    with open('../IS_final_project/models/decision_tree.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    
    # ฝึก KMeans Clustering บนข้อมูลทั้งหมด (unsupervised)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # บันทึกโมเดล KMeans
    with open('../IS_final_project/models/kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    print("Dataset1 models trained and saved.")

def train_dataset2_model():
    # เตรียมข้อมูล Dataset2
    df2 = prepare_dataset2(file_path='../IS_final_project/data/dataset2.csv')
    X2 = df2[['Sensor1', 'Sensor2', 'Sensor3']]
    y2 = df2['Output']
    
    from sklearn.model_selection import train_test_split
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    
    # สร้าง Neural Network สำหรับ Regression
    nn_model = Sequential([
        Dense(64, input_dim=3, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    nn_model.compile(loss='mean_squared_error', optimizer='adam')
    
    history = nn_model.fit(X2_train, y2_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    loss = nn_model.evaluate(X2_test, y2_test)
    print("Neural Network Test Loss:", loss)
    
    # บันทึกโมเดล Neural Network
    nn_model.save('../IS_final_project/models/neural_network.h5')
    print("Dataset2 model trained and saved.")

if __name__ == '__main__':
    train_dataset1_models()
    train_dataset2_model()
