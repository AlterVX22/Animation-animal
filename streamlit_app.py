import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import tempfile
from model_utils import get_model, idx_to_name


st.set_page_config(page_title="Animal Detector", layout="wide")
st.title("🎬 Обнаружение животных на видео")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

uploaded_video = st.file_uploader("📤 Загрузите видео (.mp4)", type=["mp4"])

if uploaded_video is not None:
    st.video(uploaded_video)

    st.info("⏳ Обрабатываю видео по кадрам...")
    
    # Временное сохранение
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stframe = st.empty()
    progress = st.progress(0, text="Обработка видео...")

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертация кадра → Tensor
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(input_tensor)[0]

        # Отрисовка предсказаний
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score >= 0.5:
                x1, y1, x2, y2 = map(int, box.tolist())
                class_name = idx_to_name.get(label.item(), "unknown")
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stframe.image(frame, channels="BGR", use_container_width=True)  # <-- Исправлено здесь

        current_frame += 1
        progress.progress(min(current_frame / total_frames, 1.0), text=f"Кадр {current_frame}/{total_frames}")

    cap.release()
    progress.empty()
    st.success("✅ Обработка завершена.")