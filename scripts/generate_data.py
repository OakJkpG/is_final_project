# generate_data.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

# ตั้งค่า seed สำหรับความ reproducible
np.random.seed(42)

##############################################
# ส่วนที่ 1: สร้าง ML Dataset (Health & Financial Data)
##############################################

# จำนวนข้อมูลสำหรับ ML dataset
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

# บันทึก ML datasets เป็นไฟล์ CSV
health_df.to_csv('data/health_dataset.csv', index=False)
financial_df.to_csv('data/financial_dataset.csv', index=False)

print("ML Datasets generated: 'data/health_dataset.csv' and 'data/financial_dataset.csv'")


##############################################
# ส่วนที่ 2: สร้าง NN Dataset สำหรับ CNN (Synthetic Digit Images)
##############################################

# จำนวน synthetic image ที่ต้องการสร้าง
n_images = 50

# ขนาด canvas ของ synthetic image
canvas_size = (64, 64)

# โฟลเดอร์เก็บข้อมูลสำหรับ NN dataset
output_dir = "data/digits"
image_dir = os.path.join(output_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# โหลด MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()  # ใช้เฉพาะส่วนทดสอบเป็นฐานตัวอย่าง
# ปรับขนาดภาพ MNIST ให้อยู่ในรูปแบบ PIL Image
mnist_images = [Image.fromarray(img) for img in x_test]
mnist_labels = y_test

# รายการสำหรับเก็บข้อมูล label ของ synthetic images
records = []

for i in range(n_images):
    # สร้าง canvas สีขาว
    canvas = Image.new("L", canvas_size, color=255)  # โหมด L สำหรับ grayscale, 255 = white
    # จำนวนตัวเลขที่จะวางในภาพนี้ (สุ่ม 3 ถึง 7 ตัว)
    n_digits = np.random.randint(3, 8)
    
    # สร้าง label vector จำนวน 10 ค่า สำหรับตัวเลข 0-9
    label_vector = np.zeros(10, dtype=int)
    
    for _ in range(n_digits):
        # เลือกดิจิทสุ่มจาก MNIST
        idx = np.random.randint(0, len(mnist_images))
        digit_img = mnist_images[idx]
        digit_label = int(mnist_labels[idx])
        # ปรับขนาดดิจิทให้เล็กลง (เช่น 14x14) เพื่อให้สามารถวางได้หลายตัวใน canvas 64x64
        digit_img = digit_img.resize((14, 14))
        
        # เลือกตำแหน่งสุ่มบน canvas ให้แน่ใจว่า digit จะไม่ออกนอกขอบ
        max_x = canvas_size[0] - 14
        max_y = canvas_size[1] - 14
        pos_x = np.random.randint(0, max_x+1)
        pos_y = np.random.randint(0, max_y+1)
        
        # วาง digit ลงบน canvas
        canvas.paste(digit_img, (pos_x, pos_y))
        
        # อัปเดท label vector
        label_vector[digit_label] += 1
    
    # บันทึกภาพ
    filename = f"digit_{i:03d}.png"
    canvas.save(os.path.join(image_dir, filename))
    
    # เก็บข้อมูลใน records
    records.append({
        "filename": filename,
        "label": label_vector.tolist()  # แปลง numpy array เป็น list
    })

# บันทึก labels เป็น CSV สำหรับ NN dataset
df_labels = pd.DataFrame(records)
df_labels.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

print(f"Generated {n_images} synthetic digit images and saved labels to {os.path.join(output_dir, 'labels.csv')}")
