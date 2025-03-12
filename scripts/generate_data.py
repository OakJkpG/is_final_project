import os
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

# ตั้งค่า seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รันโค้ด
np.random.seed(42)

##############################################
# ส่วนที่ 1: สร้าง ML Dataset (Financial Data)
##############################################
n_financial = 200

# สร้าง Financial Dataset
financial_df = pd.DataFrame({
    'ID': np.arange(1, n_financial + 1),
    'StockPrice': np.random.normal(200, 30, n_financial),
    'Income': np.random.normal(75000, 10000, n_financial),
    'Expense': np.random.normal(50000, 8000, n_financial)
})

# คำนวณ NetProfit = Income - Expense + noise
noise = np.random.normal(0, 2000, n_financial)
financial_df['NetProfit'] = financial_df['Income'] - financial_df['Expense'] + noise

# แทรก missing values (10%) ในฟีเจอร์ StockPrice, Income, Expense
for col in ['StockPrice', 'Income', 'Expense']:
    mask = np.random.rand(n_financial) < 0.1
    financial_df.loc[mask, col] = np.nan

# สร้างโฟลเดอร์ data หากยังไม่มี
if not os.path.exists('data'):
    os.makedirs('data')

# บันทึก Financial Dataset เป็นไฟล์ CSV
financial_df.to_csv('data/financial_dataset.csv', index=False)

##############################################
# ส่วนที่ 2: สร้าง NN Dataset สำหรับ CNN (Synthetic Digit Images)
##############################################
n_images = 50
canvas_size = (64, 64)

# folder เก็บข้อมูลสำหรับ NN dataset
output_dir = "data/digits"
image_dir = os.path.join(output_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# โหลด MNIST dataset (ใช้เฉพาะส่วนทดสอบเป็นฐานตัวอย่าง)
(_, _), (x_test, y_test) = mnist.load_data()
mnist_images = [Image.fromarray(img) for img in x_test]
mnist_labels = y_test

# รายการสำหรับเก็บข้อมูล label ของ synthetic images
records = []

for i in range(n_images):
    # สร้าง canvas สีขาว (โหมด L สำหรับ grayscale)
    canvas = Image.new("L", canvas_size, color=255)
    n_digits = np.random.randint(3, 8)  # จำนวนตัวเลขที่จะวางในภาพ (สุ่ม 3 ถึง 7 ตัว)
    label_vector = np.zeros(10, dtype=int)
    
    for _ in range(n_digits):
        # เลือก digit สุ่มจาก MNIST
        idx = np.random.randint(0, len(mnist_images))
        digit_img = mnist_images[idx]
        digit_label = int(mnist_labels[idx])
        digit_img = digit_img.resize((14, 14))  # ปรับขนาดเป็น 14x14
        
        # เลือกตำแหน่งสุ่มบน canvas (ไม่ให้ออกนอกขอบ)
        max_x = canvas_size[0] - 14
        max_y = canvas_size[1] - 14
        pos_x = np.random.randint(0, max_x + 1)
        pos_y = np.random.randint(0, max_y + 1)
        
        # วาง digit ลงบน canvas
        canvas.paste(digit_img, (pos_x, pos_y))
        label_vector[digit_label] += 1

    # จำลองความไม่สมบูรณ์ของ label ด้วยการสุ่มตัด label บางครั้ง (20%)
    if np.random.rand() < 0.2:
        record = {
            "filename": f"digit_{i:03d}.png",
            "label": None  # ไม่มี label สำหรับภาพนี้
        }
    else:
        record = {
            "filename": f"digit_{i:03d}.png",
            "label": label_vector.tolist()
        }
    
    # บันทึกภาพ
    filename = f"digit_{i:03d}.png"
    canvas.save(os.path.join(image_dir, filename))
    records.append(record)

# บันทึก labels เป็น CSV สำหรับ NN dataset
df_labels = pd.DataFrame(records)
df_labels.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

print("ML Datasets generated: 'data/financial_dataset.csv' and 'data/digits/labels.csv'")
