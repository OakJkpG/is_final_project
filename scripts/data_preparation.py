# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_health_data(input_path="data/health_dataset.csv", output_path="data/health_dataset_prepared.csv"):
    """
    โหลดและเตรียมข้อมูล Health Data
    - เติม missing values สำหรับ BMI, BloodPressure, และ HeartRate ด้วยค่าเฉลี่ย
    - เข้ารหัส target 'Risk' เป็นตัวเลขด้วย LabelEncoder
    """
    df = pd.read_csv(input_path)
    
    # เติม missing values สำหรับฟีเจอร์ numeric
    for col in ['BMI', 'BloodPressure', 'HeartRate']:
        df[col].fillna(df[col].mean(), inplace=True)
    
    # เข้ารหัส target Risk
    le = LabelEncoder()
    df['Risk_enc'] = le.fit_transform(df['Risk'])
    
    # บันทึกข้อมูลที่เตรียมไว้ (optional)
    df.to_csv(output_path, index=False)
    
    return df, le

def prepare_financial_data(input_path="data/financial_dataset.csv", output_path="data/financial_dataset_prepared.csv"):
    """
    โหลดและเตรียมข้อมูล Financial Data
    - เติม missing values สำหรับ StockPrice, Income, และ Expense ด้วยค่าเฉลี่ย
    """
    df = pd.read_csv(input_path)
    
    for col in ['StockPrice', 'Income', 'Expense']:
        df[col].fillna(df[col].mean(), inplace=True)
    
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    health_df, le_health = prepare_health_data()
    print("Health data prepared and saved to data/health_dataset_prepared.csv")
    
    financial_df = prepare_financial_data()
    print("Financial data prepared and saved to data/financial_dataset_prepared.csv")
