import streamlit as st
import requests
import io

st.set_page_config(page_title="Animal Detector", layout="wide")
st.title("🎬 Обнаружение животных на видео (YOLOv8)")


try:
    response = requests.get("http://0.0.0.0:5100/status", timeout=5)

    try:
        # Проверка устройства
        st.markdown(f"**Устройство для инференса:** {requests.get('http://0.0.0.0:5100/device').json()['device name']}")

        uploaded_video = st.file_uploader("📤 Загрузите видео (.mp4)", type=["mp4"])

        if uploaded_video and st.button("Обработать видео"):
            st.video(uploaded_video)
            st.info("⏳ Обрабатываю видео по кадрам...")
            
            # Обрабатываем файл для подачи в POST
            files = {
                    "file": (uploaded_video.name, uploaded_video, "video/mp4")
                }
            response = requests.post('http://0.0.0.0:5100/upload_video', files=files) # POST 
            
            # Обработка успешности 
            if response.status_code == 200:
                 video_bytes = io.BytesIO(response.content)
                 
                 st.download_button(
                                    label="📥 Скачать обработанное видео",
                                    data=video_bytes,
                                    file_name="processed_video.mp4",
                                    mime="video/mp4"
                                    )
            else:
                st.error("Ошибка при обработке видео")
            
            



    except Exception as e:
        st.error(f'Неизвестная ошибка: {e}')

except:
    st.error('Пробелмы с подключением к API')


