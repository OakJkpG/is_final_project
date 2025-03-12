# IS_Final_Project

## Overview
โปรเจคนี้เป็นตัวอย่างการสร้างโมเดล Machine Learning และ Neural Network ด้วยข้อมูลที่มีความไม่สมบูรณ์ โดยแบ่งออกเป็น 2 ชุดข้อมูล:
- **Dataset1:** การจำแนกประเภทและการจัดกลุ่ม (Decision Tree & K-Means)
- **Dataset2:** การจำแนกรูป (CNN)

## การติดตั้งและรันโปรเจค
1. **Clone repository และสร้าง Virtual Environment**  
   git clone <repository_url>
   cd IS_final_project
   python -m venv venv
   source venv/bin/activate
   venv\Scripts\activate
   pip install -r requirements.txt
2. **สร้าง Dataset**
    cd scripts
    python generate_data.py
3. **ฝึก model**
    python model_training.py
4. **รันเว็บ**
    streamlit run app.py หรือ python -m streamlit run app.py

Python Version 3.12.9