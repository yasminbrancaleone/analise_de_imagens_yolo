from ultralytics import YOLO
import numpy as np
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

caminho_video = os.path.join(OUTPUT_DIR, "ex7_invasao.mp4")
caminho_print = os.path.join(OUTPUT_DIR, "ex7_print_invasao.jpg")
out = cv2.VideoWriter(caminho_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))

cv2.namedWindow("Detector de Invasao (ROI)", cv2.WINDOW_NORMAL)
invasao_salva = False
ultimo_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    poligono = np.array([
        [int(largura * 0.25), int(altura * 0.45)],
        [int(largura * 0.75), int(altura * 0.45)],
        [int(largura * 0.85), int(altura * 0.90)],
        [int(largura * 0.15), int(altura * 0.90)]
    ], dtype=np.int32)

    results = model.predict(source=frame, conf=0.5, classes=[0], verbose=False)
    res = results[0]
    annotated_frame = res.plot()

    invasao_detectada = False

    if res.boxes is not None:
        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(annotated_frame, (cx, cy), 5, (255, 255, 0), -1)

            dentro = cv2.pointPolygonTest(poligono, (cx, cy), False)
            if dentro >= 0:
                invasao_detectada = True

    cor_poligono = (0, 0, 255) if invasao_detectada else (0, 255, 0)
    cv2.polylines(annotated_frame, [poligono], isClosed=True, color=cor_poligono, thickness=3)

    if invasao_detectada:
        cv2.rectangle(annotated_frame, (10, 10), (470, 60), (0, 0, 255), -1)
        cv2.putText(annotated_frame, "ALERTA: INVASAO DE AREA PROIBIDA!", (20, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if not invasao_salva:
            cv2.imwrite(caminho_print, annotated_frame)
            invasao_salva = True

    out.write(annotated_frame)
    ultimo_frame = annotated_frame
    cv2.imshow("Detector de Invasao (ROI)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if not invasao_salva and ultimo_frame is not None:
    cv2.imwrite(caminho_print, ultimo_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo salvo em: {caminho_video}\nPrint salvo em: {caminho_print}")