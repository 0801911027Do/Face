# pyrefly: ignore [missing-import]
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 1. เชื่อมต่อกล้อง
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# 2. รับชื่อผู้ใช้
name = ""
while not name:
    name = input('name: ').strip()

user_dir = os.path.join(DATA_DIR, name)
os.makedirs(user_dir, exist_ok=True)

# นับลำดับรูปต่อจากรูปเดิม (ถ้ามี) เพื่อไม่ให้เซฟทับ
existing_files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
i = len(existing_files) + 1

print("\n" + "=" * 50)
print(f" พร้อมบันทึกภาพสำหรับ: {name} (มีอยู่แล้ว {len(existing_files)} รูป)")
print(" [คำสั่ง] กด 's' เพื่อบันทึกภาพใบหน้า | กด 'q' เพื่อออก")
print("=" * 50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # กรอบสี่เหลี่ยมและตัดภาพใบหน้าแบบดิบ (Raw Pixels 180x140)
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 0, 255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390], cv2.COLOR_BGR2GRAY)
    
    cv2.putText(frame, f"User: {name} | Saved: {i - 1}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Press 's': Save | 'q': Quit", (15, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

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
print(f"เสร็จสิ้น บันทึกรูปทั้งหมดสำหรับ {name} รวม {i - 1} รูป")