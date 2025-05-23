# Animation-animal

## Задача детекции диких животных с использованием YOLOv8n

Набор данных и аннотации изображений находятся на сайте [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)

Использовались файлы:
- Benchmark images (6GB)
- Metadata files for train/val/cis/trans splits (3MB)

Для запуска кода скачайте необходимые библиотеки (если у вас доступна CUDA):
`pip install -r requirements_Kovalev.txt`

Для запуска выполните следующие команды:

```
python model_api.py
streamlit run streamlit_yolo.py
```

#### Пример работы сайта (при запуске streamlit_yolo_app.py)
![Пример](https://github.com/AlterVX22/Animation-animal/blob/main/animation.gif)
