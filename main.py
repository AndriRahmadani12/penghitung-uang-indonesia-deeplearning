import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('src', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'coba22.mp4')
video_path_out = '{}_out2.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.8

total_money = {'RP1000': 0, 'RP2000': 0, 'RP5000': 0, 'RP10000': 0, 'RP20000': 0, 'RP50000': 0, 'RP100000': 0}

import locale

# Set locale ke bahasa Indonesia untuk format Rupiah
locale.setlocale(locale.LC_ALL, 'id_ID')

while ret:
    # Reset total nilai uang untuk setiap frame
    total_frame_money = 0

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            class_name = results.names[int(class_id)].upper()
            # Jika kelas adalah salah satu dari label uang, tambahkan nilainya ke total_frame_money
            if class_name in total_money:
                total_money[class_name] += 1
                total_frame_money += int(class_name[2:])  # Tambahkan nilai uang ke total_frame_money

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)


    total_frame_money_str = locale.currency(total_frame_money, grouping=True)

    # Tampilkan total uang terdeteksi di frame video
    cv2.putText(frame, f"Total uang: {total_frame_money_str}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

