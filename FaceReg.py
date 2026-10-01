# pyrefly: ignore [missing-import]
import cv2
import os
import numpy as np

# ==============================================================
# 1. ฟังก์ชัน Machine Learning: K-Nearest Neighbors (KNN) บน Raw Pixels
# ==============================================================
def knn(X, y, z, k=3):
    # คำนวณ Euclidean Distance บน float32 เพื่อความถูกต้องทางคณิตศาสตร์
    diff = X - z
    d = np.sum(diff ** 2, axis=1)
    idx = np.argsort(d)[:k]
    
    # Majority Voting (โหวตชื่อตามเสียงข้างมากของ k เพื่อนบ้านที่ใกล้ที่สุด)
    top_labels = y[idx]
    vals, counts = np.unique(top_labels, return_counts=True)
    return vals[np.argmax(counts)]

# ==============================================================
# 2. โหลด Dataset ทั้งหมด (แปลงภาพเป็น 1D Vector ดิบ 180x140 = 25,200)
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

X, y = [], []
if not os.path.exists(DATA_DIR):
    print(f"Error: ไม่พบโฟลเดอร์ {DATA_DIR}")
    exit(1)

print("\nกำลังโหลด Dataset...")
for name in sorted(os.listdir(DATA_DIR)):
    p_path = os.path.join(DATA_DIR, name)
    if os.path.isdir(p_path):
        count = 0
        for img in sorted(os.listdir(p_path)):
            img_file = os.path.join(p_path, img)
            img_data = cv2.imread(img_file, 0)
            if img_data is not None:
                X.append(img_data.flatten())
                y.append(name)
                count += 1
        print(f" - {name}: {count} รูป")

if len(X) == 0:
    print("ไม่พบรูปภาพใน dataset กรุณารัน FaceTrain.py เพื่อเก็บภาพใบหน้าก่อนครับ")
    exit(1)

X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f"โหลดข้อมูลเสร็จสิ้น: รวม {len(X)} ตัวอย่าง, Feature Vector ขนาด {X.shape[1]}\n")

# ==============================================================
# 3. อ่านกล้องและทำนายผลด้วยโมเดล KNN แบบ Real-time
# ==============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("เริ่มระบบจดจำใบหน้า (กด 'q' เพื่อออกจากโปรแกรม)")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # ตัดภาพใบหน้าขนาด 180x140 ตามกรอบสี่เหลี่ยม (Raw Pixels)
    face = cv2.cvtColor(frame[120:300, 250:390], cv2.COLOR_BGR2GRAY)
    
    # ส่ง 1D Vector เข้าฟังก์ชัน KNN ทำนายชื่อ
    label = knn(X, y, face.flatten().astype(np.float32))

    # แสดงผลบนหน้าจอ
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 255, 0), 2)
    cv2.putText(frame, label, (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    cv2.imshow('face', face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
