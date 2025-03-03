import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# นำเข้า module สำหรับเตรียมข้อมูล (เฉพาะ Financial Data)
from data_preparation import prepare_financial_data, prepare_digit_images_data

##############################
# 1. Training Financial Data with Decision Tree and KMeans
##############################
def train_financial_dt_kmeans():
    # เตรียมข้อมูล Financial Data
    df_financial = prepare_financial_data()
    
    # สร้าง target ใหม่สำหรับการจำแนก:
    # กำหนดให้ Profit_Class = 1 หาก NetProfit >= median, 0 หากต่ำกว่า
    median_profit = df_financial['NetProfit'].median()
    df_financial['Profit_Class'] = (df_financial['NetProfit'] >= median_profit).astype(int)
    
    # เลือกฟีเจอร์และ target สำหรับ Decision Tree
    X = df_financial[['StockPrice', 'Income', 'Expense']]
    y = df_financial['Profit_Class']
    
    # แบ่งข้อมูลเป็น training และ testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ฝึกโมเดล Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # ประเมินโมเดลด้วยข้อมูล test
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy on Financial Data: {accuracy:.2f}")
    
    # สร้างโฟลเดอร์ models หากยังไม่มี
    os.makedirs("models", exist_ok=True)
    
    # บันทึกโมเดล Decision Tree
    with open("models/financial_decision_tree.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    
    # ฝึกโมเดล KMeans สำหรับ clustering บนฟีเจอร์เดียวกัน
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(X)
    
    # บันทึกโมเดล KMeans
    with open("models/financial_kmeans.pkl", "wb") as f:
        pickle.dump(kmeans_model, f)
    
    print("Financial Decision Tree and KMeans models saved in 'models/' directory.")
    print("Cluster labels (first 10 samples):", kmeans_model.labels_[:10])

##############################
# 2. Training Synthetic Image Data with CNN (Regression for Digit Counting)
##############################
def train_digit_cnn():
    # ก่อนเริ่มฝึก CNN ให้เตรียมข้อมูล label ใหม่ (preparation)
    prepare_digit_images_data()  # สร้างไฟล์ labels_prepared.csv
    # กำหนด path สำหรับ dataset
    data_dir = "data/digits"
    image_dir = os.path.join(data_dir, "images")
    labels_csv = os.path.join(data_dir, "labels_prepared.csv")  # ใช้ไฟล์ labels ที่เตรียมไว้แล้ว
    
    # โหลด labels
    df = pd.read_csv(labels_csv)
    # แปลงคอลัมน์ label จาก string เป็น list (ใช้ eval)
    df["label"] = df["label"].apply(eval)
    labels = np.stack(df["label"].values)
    
    # เตรียมข้อมูลภาพ
    images = []
    for fname in df["filename"].values:
        img_path = os.path.join(image_dir, fname)
        img = load_img(img_path, color_mode="grayscale", target_size=(64, 64))
        img = img_to_array(img) / 255.0  # normalize
        images.append(img)
    images = np.array(images)
    
    # แบ่งข้อมูลเป็น training/test set
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # สร้างโมเดล CNN สำหรับ regression (นับจำนวนตัวเลข 0-9)
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(64,64,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="linear")  # output vector สำหรับนับตัวเลข 0-9
    ])
    
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    model.summary()
    
    # ฝึกโมเดล
    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)
    
    # ประเมินโมเดล
    loss = model.evaluate(X_test, y_test)
    print(f"Digit Counting CNN Test Loss: {loss:.2f}")
    
    # บันทึกโมเดล CNN
    os.makedirs("models", exist_ok=True)
    model.save("models/digit_count_cnn.h5")
    print("CNN model for digit counting saved as 'models/digit_count_cnn.h5'.")

if __name__ == "__main__":
    train_financial_dt_kmeans()
    train_digit_cnn()
