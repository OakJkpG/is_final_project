# generate_data.py
import os
import pandas as pd
import numpy as np

# ตั้งค่า seed สำหรับความ reproducible
np.random.seed(42)

# จำนวนข้อมูล
n_health = 200
n_financial = 200

# สร้าง Health Data Dataset
health_df = pd.DataFrame({
    'ID': np.arange(1, n_health + 1),
    'BMI': np.random.normal(25, 4, n_health),
    'BloodPressure': np.random.normal(120, 15, n_health),
    'HeartRate': np.random.normal(75, 10, n_health),
    'Risk': np.random.choice(['Low', 'Medium', 'High'], size=n_health, p=[0.4, 0.4, 0.2])
})

# แทรก missing values (~10%) ในฟีเจอร์ BMI, BloodPressure, HeartRate
for col in ['BMI', 'BloodPressure', 'HeartRate']:
    mask = np.random.rand(n_health) < 0.1
    health_df.loc[mask, col] = np.nan

# สร้าง Financial Data Dataset
financial_df = pd.DataFrame({
    'ID': np.arange(1, n_financial + 1),
    'StockPrice': np.random.normal(200, 30, n_financial),
    'Income': np.random.normal(75000, 10000, n_financial),
    'Expense': np.random.normal(50000, 8000, n_financial)
})
# คำนวณ NetProfit = Income - Expense + noise
noise = np.random.normal(0, 2000, n_financial)
financial_df['NetProfit'] = financial_df['Income'] - financial_df['Expense'] + noise

# แทรก missing values (~10%) ในฟีเจอร์ StockPrice, Income, Expense
for col in ['StockPrice', 'Income', 'Expense']:
    mask = np.random.rand(n_financial) < 0.1
    financial_df.loc[mask, col] = np.nan

# สร้างโฟลเดอร์ data หากยังไม่มี
if not os.path.exists('data'):
    os.makedirs('data')

# บันทึกเป็นไฟล์ CSV
health_df.to_csv('data/health_dataset.csv', index=False)
financial_df.to_csv('data/financial_dataset.csv', index=False)

print("Datasets generated: 'data/health_dataset.csv' and 'data/financial_dataset.csv'")
