import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import torch
import warnings
import random

warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # для подавления TensorFlow

st.set_page_config(page_title="Animal Detector", layout="wide")
st.title("🎬 Обнаружение животных на видео (YOLOv8)")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("weights/best.pt")  # путь к твоей модели

if device == "cuda":
    device_name = torch.cuda.get_device_name(0)
else:
    device_name = "CPU"



colors = {}
for cls_id, cls_name in model.names.items():
    colors[cls_id] = tuple(random.choices(range(256), k=3))

st.markdown(f"**Устройство для инференса:** {device_name}")


uploaded_video = st.file_uploader("📤 Загрузите видео (.mp4)", type=["mp4"])

if uploaded_video is not None:
    st.video(uploaded_video)

    st.info("⏳ Обрабатываю видео по кадрам...")

    # Временный входной файл
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Временный выходной файл
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = temp_out.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    stframe = st.empty()
    progress = st.progress(0, text="Обработка видео...")

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, device=device, save=False, conf=0.5, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            color = colors.get(cls_id, (0, 255, 0))  # цвет по классу, если не нашли — зелёный

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)  # сохраняем в видео
        stframe.image(frame, channels="BGR", use_container_width=True)

        current_frame += 1
        progress.progress(min(current_frame / total_frames, 1.0), text=f"Кадр {current_frame}/{total_frames}")

    cap.release()
    out.release()
    progress.empty()
    st.success("✅ Обработка завершена.")

    # Загрузка
    with open(out_path, 'rb') as f:
        st.download_button("📥 Скачать видео с детекцией", f, file_name="result.mp4", mime="video/mp4")
