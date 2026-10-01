import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 1. เชื่อมต่อกล้อง
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# 2. รับชื่อผู้ใช้และเตรียมโฟลเดอร์เก็บข้อมูล
name = input('name: ').strip()
while not name:
    name = input('name: ').strip()

user_dir = os.path.join(DATA_DIR, name)
os.makedirs(user_dir, exist_ok=True)

# ตรวจสอบรูปภาพเดิมเพื่อกำหนดลำดับรูปถัดไป
existing_files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
i = len(existing_files) + 1

# 3. เริ่มบันทึกภาพใบหน้า
print("กด 's' เพื่อบันทึกภาพ | กด 'q' เพื่อออก")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # วาดกรอบและตัดภาพใบหน้า
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 0, 255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390], cv2.COLOR_BGR2GRAY)

    # แสดงชื่อและจำนวนรูปบนหน้าจอ
    cv2.putText(frame, f"User: {name} | Saved: {i - 1}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('face', face)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_path = os.path.join(user_dir, f'{name}_{i}.jpg')
        cv2.imwrite(img_path, face)
        print(f"บันทึกรูปที่ {i}: {img_path}")
        i += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()