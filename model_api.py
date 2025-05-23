from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile
import cv2
import os
import torch
from ultralytics import YOLO
import random

app = FastAPI()

# БЛОК ИНИЦИАЛИЗАЦИИ МОДЕЛИ И УСТРОЙСТВА

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # для подавления TensorFlow

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("weights/best.pt")  # путь к модели

if device == "cuda":
    device_name = torch.cuda.get_device_name(0)
else:
    device_name = "CPU"

# Генерация цветов для классов
colors = {}
for cls_id, cls_name in model.names.items():
    colors[cls_id] = tuple(random.choices(range(256), k=3))

@app.get('/status')
def get_status():
    return {'status': 'ok'}

@app.get('/device')
def get_device():
    return {'device name': device_name}

@app.post('/upload_video')
def upload_video(file: UploadFile = File(...)):
    # Считываем содержимое файла в байты
    contents = file.file.read()
    
    # БЛОК СОЗДАНИЯ ВРЕМЕННОГО ФАЙЛА
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(contents)
    tfile.close()  # обязательно закрываем файл
    video_path = tfile.name  # путь к временно сохранённому видео     
    
    # БЛОК ОТКРЫТИЯ ВИДЕО ЧЕРЕЗ OpenCV
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Временный выходной файл для записи результата
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = temp_out.name
    
    # Настройка видеокодека и объекта записи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # БЛОК ОБРАБОТКИ ВИДЕО
    current_frame = 0
    while cap.isOpened():
    
        ret, frame = cap.read() # считывание видео по кадрам
        results = model.predict(source=frame, device=device, save=False, conf=0.5, verbose=False)[0]
        if not ret:
            break
        
        for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                color = colors.get(cls_id, (0, 255, 0))  # цвет по классу, если не нашли — зелёный

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        out.write(frame)
        
        current_frame += 1

    cap.release()
    out.release()
    print(f"[DEBUG] Обработано кадров: {current_frame}")
    print(f"[DEBUG] Файл существует? {os.path.exists(out_path)}")
    print(f"[DEBUG] Размер выходного видео: {os.path.getsize(out_path)} байт")

    
    return FileResponse(out_path, media_type="video/mp4", filename="processed_video.mp4")
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5100)
    