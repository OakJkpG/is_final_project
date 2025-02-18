# My IS_final_project Machine Learning Project

## Overview
โปรเจคนี้เป็นตัวอย่างการสร้างโมเดล Machine Learning ด้วยข้อมูลที่มีความไม่สมบูรณ์ โดยแบ่งออกเป็น 2 ชุดข้อมูล:
- **Dataset1:** สำหรับการจำแนกประเภท (Decision Tree) และการจัดกลุ่ม (K-Means)
- **Dataset2:** สำหรับการทำนายค่า (Regression) ด้วย Neural Network

นอกจากนี้ยังพัฒนาเว็บแอปด้วย Streamlit เพื่อแสดงขั้นตอนการเตรียมข้อมูล การพัฒนาโมเดล และสาธิตการทำงานของโมเดล

## โครงสร้างโปรเจค
my_streamlit_project/ 
├── data/
├── models/
├── scripts/
│ ├── generate_data.py
│ ├── data_preparation.py
│ └── model_training.py
├── app.py
├── requirements.txt
└── README.md


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
    streamlit run app.py


