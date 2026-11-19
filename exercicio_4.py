from ultralytics import YOLO
from collections import Counter
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

video_source = os.path.join(BASE_DIR, "video.mp4")  # Ou 0 para webcam
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Erro ao abrir a fonte de vídeo.")
    exit()

largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

caminho_video = os.path.join(OUTPUT_DIR, "ex4_contador.mp4")
caminho_print = os.path.join(OUTPUT_DIR, "ex4_print.jpg")
out = cv2.VideoWriter(caminho_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))

cv2.namedWindow("Contador Dinamico", cv2.WINDOW_NORMAL)
print_salvo = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, classes=[0, 2], verbose=False)
    res = results[0]
    annotated_frame = res.plot()

    if res.boxes is not None and res.boxes.cls is not None:
        detected_classes = res.boxes.cls.int().tolist()
        contagem = Counter(detected_classes)
        total_pessoas = contagem[0]
        total_carros = contagem[2]
    else:
        total_pessoas = 0
        total_carros = 0

    cv2.rectangle(annotated_frame, (10, 10), (280, 85), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Pessoas: {total_pessoas}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Carros: {total_carros}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(annotated_frame)
    if not print_salvo and (total_pessoas + total_carros) > 0:
        cv2.imwrite(caminho_print, annotated_frame)
        print_salvo = True

    cv2.imshow("Contador Dinamico", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo salvo em: {caminho_video}\nPrint salvo em: {caminho_print}")