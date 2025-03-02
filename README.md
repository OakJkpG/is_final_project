# My IS_final_project Machine Learning Project

## Overview
โปรเจคนี้เป็นตัวอย่างการสร้างโมเดล Machine Learning ด้วยข้อมูลที่มีความไม่สมบูรณ์ โดยแบ่งออกเป็น 3 ชุดข้อมูล:
- **Dataset1:** การจำแนกประเภท (Decision Tree)
- **Dataset2:** การจัดกลุ่ม (K-Means)
- **Dataset3:** การจำแนกรูป (CNN)

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

Python Version 3.10.6
