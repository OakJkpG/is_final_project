# scripts/generate_data.py
import os
import pandas as pd
import numpy as np

# สร้างโฟลเดอร์ data หากยังไม่มี
os.makedirs('../IS_final_project/data', exist_ok=True)

# กำหนดจำนวนตัวอย่าง
n = 1000
np.random.seed(42)

# สร้าง Dataset1 สำหรับ Classification และ Clustering
data1 = pd.DataFrame({
    'ID': range(1, n+1),
    'Feature_A': np.random.normal(loc=50, scale=10, size=n),
    'Feature_B': np.random.uniform(low=0, high=100, size=n),
    'Feature_C': np.random.choice(['Low', 'Medium', 'High'], size=n)
})

# Introduce missing values (ประมาณ 10% ต่อคอลัมน์)
for col in ['Feature_A', 'Feature_B', 'Feature_C']:
    data1.loc[data1.sample(frac=0.1, random_state=42).index, col] = np.nan

# สร้าง Target แบบสุ่ม (3 คลาส)
data1['Target'] = np.random.choice(['Class1', 'Class2', 'Class3'], size=n)

# บันทึกเป็น CSV
data1.to_csv('../IS_final_project/data/dataset1.csv', index=False)


# สร้าง Dataset2 สำหรับ Regression (Neural Network)
data2 = pd.DataFrame({
    'ID': range(1, n+1),
    'Sensor1': np.random.normal(loc=20, scale=5, size=n),
    'Sensor2': np.random.normal(loc=50, scale=15, size=n),
    'Sensor3': np.random.normal(loc=100, scale=20, size=n)
})

# Introduce missing values (ประมาณ 10% ต่อคอลัมน์)
for col in ['Sensor1', 'Sensor2', 'Sensor3']:
    data2.loc[data2.sample(frac=0.1, random_state=42).index, col] = np.nan

# สร้าง Output โดยใช้ Sensor1, Sensor2 และ Sensor3 (non-linear พร้อม noise)
data2['Output'] = (
    data2['Sensor1'].fillna(data2['Sensor1'].mean()) * 0.3 +
    data2['Sensor2'].fillna(data2['Sensor2'].mean()) * 0.5 +
    np.sin(data2['Sensor3'].fillna(data2['Sensor3'].mean())/10) * 10 +
    np.random.normal(0, 2, n)
)

# บันทึกเป็น CSV
data2.to_csv('../IS_final_project/data/dataset2.csv', index=False)

print("Datasets generated successfully in the ../data folder.")
