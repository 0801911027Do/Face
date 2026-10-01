import os
import cv2
import numpy as np

# 1. ฟังก์ชัน KNN (คำนวณ Euclidean distance บน float32 และโหวตเสียงข้างมาก)
def knn(X, y, z, k=3):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    labels, counts = np.unique(y[idx], return_counts=True)
    return labels[np.argmax(counts)]

# 2. โหลดรูปภาพจากโฟลเดอร์ data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

if not os.path.exists(DATA_DIR):
    print(f"Error: ไม่พบโฟลเดอร์ {DATA_DIR}")
    exit(1)

X, y = [], []
for name in sorted(os.listdir(DATA_DIR)):
    user_path = os.path.join(DATA_DIR, name)
    if os.path.isdir(user_path):
        for img_name in sorted(os.listdir(user_path)):
            img = cv2.imread(os.path.join(user_path, img_name), 0)
            if img is not None:
                X.append(img.flatten())
                y.append(name)

if len(X) == 0:
    print("ไม่พบข้อมูลรูปภาพ กรุณารัน FaceTrain.py ก่อนครับ")
    exit(1)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# 3. เปิดกล้องและทำนายผลแบบ Real-time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("เริ่มการทำงาน (กด 'q' เพื่อออก)")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # ครอปตำแหน่งใบหน้า (พิกัด 180x140)
    face = cv2.cvtColor(frame[120:300, 250:390], cv2.COLOR_BGR2GRAY)
    
    # ทำนายผล
    label = knn(X, y, face.flatten().astype(np.float32))

    # วาดกรอบและแสดงชื่อ
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 255, 0), 2)
    cv2.putText(frame, label, (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    cv2.imshow('face', face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
