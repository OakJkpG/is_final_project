import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ast

# โหลดและเตรียมข้อมูล Financial Data
def prepare_financial_data(input_path="data/financial_dataset.csv", output_path="data/financial_dataset_prepared.csv"):
    # เติม missing values สำหรับ StockPrice, Income, และ Expense ด้วยค่าเฉลี่ย
    df = pd.read_csv(input_path)
    for col in ['StockPrice', 'Income', 'Expense']:
        df[col].fillna(df[col].mean(), inplace=True)
    
    df.to_csv(output_path, index=False)
    return df

# โหลดและเตรียมข้อมูล Synthetic Digit Images Data
def prepare_digit_images_data(input_path="data/digits/labels.csv", output_path="data/digits/labels_prepared.csv"):
    # ตรวจสอบและจัดการกับ missing labels โดยแทนที่ด้วยเวกเตอร์ [0]*10
    # แปลงค่าในคอลัมน์ 'label' จากสตริงเป็นลิสต์ของตัวเลข
    
    df = pd.read_csv(input_path)
    
    def parse_label(label):
        # ถ้าเป็น missing หรือเป็น 'None' ให้แทนด้วยเวกเตอร์ 0
        if pd.isna(label) or label == 'None':
            return [0] * 10
        try:
            parsed = ast.literal_eval(label)
            if isinstance(parsed, list):
                return parsed
            else:
                return [0] * 10
        except:
            return [0] * 10

    df['label'] = df['label'].apply(parse_label)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    financial_df = prepare_financial_data()
    print("สร้าง financial_dataset_prepared.csv สำเร็จ")
    
    digit_df = prepare_digit_images_data()
    print("สร้าง labels_prepared.csv สำเร็จ")
