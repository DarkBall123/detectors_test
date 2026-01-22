Проект сравнивает две архитектуры детекции людей на одном видеофайле. Используются RT DETR и YOLOv8. Пайплайн чтения видео, отрисовки и сохранения общий.

Требования. Нужны Python 3.10 или новее, pip, и ffmpeg или OpenCV с поддержкой mp4.

Подготовка видео. После клонирования репозитория положите файл crowd.mp4 в папку artifacts. В командах дальше используется путь artifacts/crowd.mp4.

Установка. Выполните команды по очереди.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Запуск RT DETR.
python3 -m app.main --model rtdetr --input artifacts/crowd.mp4 --output outputs/crowd_rtdetr.mp4 --device cpu --conf 0.5 --save-json outputs/metrics_rtdetr.json

Запуск YOLOv8.
python3 -m app.main --model yolov8 --input artifacts/crowd.mp4 --output outputs/crowd_yolov8.mp4 --device cpu --conf 0.4 --save-json outputs/metrics_yolov8.json

Видео сохраняются в outputs. Метрики сохраняются в outputs/metrics_rtdetr.json и outputs/metrics_yolov8.json. Отчет находится в reports/report.md.

ONNX. Экспорт RT DETR.
python3 -m tools.export_onnx --weights rtdetr-l.pt --output artifacts/rtdetr.onnx --opset 16

ONNX инференс.
python3 -m tools.infer_onnx --model artifacts/rtdetr.onnx --input artifacts/crowd.mp4 --output outputs/crowd_rtdetr_onnx.mp4 --conf 0.5 --save-json outputs/metrics_rtdetr_onnx.json
