# scripts/data_preparation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_dataset1(file_path='../IS_final_project/data/dataset1.csv'):
    df1 = pd.read_csv(file_path)
    
    # เติม missing values สำหรับตัวเลขด้วยค่าเฉลี่ย
    df1['Feature_A'].fillna(df1['Feature_A'].mean(), inplace=True)
    df1['Feature_B'].fillna(df1['Feature_B'].mean(), inplace=True)
    
    # เติม missing values สำหรับตัวแปร categorical ด้วย mode
    df1['Feature_C'].fillna(df1['Feature_C'].mode()[0], inplace=True)
    
    # เข้ารหัส Feature_C เป็นตัวเลข
    le_feature = LabelEncoder()
    df1['Feature_C_enc'] = le_feature.fit_transform(df1['Feature_C'])
    
    # เข้ารหัส Target สำหรับ Classification
    le_target = LabelEncoder()
    df1['Target_enc'] = le_target.fit_transform(df1['Target'])
    
    return df1, le_feature, le_target

def prepare_dataset2(file_path='../IS_final_project/data/dataset2.csv'):
    df2 = pd.read_csv(file_path)
    
    # เติม missing values สำหรับ Sensor columns ด้วยค่าเฉลี่ย
    for col in ['Sensor1', 'Sensor2', 'Sensor3']:
        df2[col].fillna(df2[col].mean(), inplace=True)
        
    return df2
