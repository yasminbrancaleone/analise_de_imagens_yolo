from ultralytics import YOLO
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

caminho_video = os.path.join(OUTPUT_DIR, "ex5_pedagio.mp4")
caminho_print = os.path.join(OUTPUT_DIR, "ex5_print.jpg")
out = cv2.VideoWriter(caminho_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))

cv2.namedWindow("Pedagio Virtual Bidirecional", cv2.WINDOW_NORMAL)

historico_y = {}
ids_processados = set()
sentido_a = 0
sentido_b = 0
ultimo_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    linha_y = altura // 2

    results = model.track(source=frame, conf=0.5, classes=[2, 3, 5, 7], persist=True, verbose=False)
    res = results[0]
    annotated_frame = res.plot()

    cv2.line(annotated_frame, (0, linha_y), (largura, linha_y), (255, 0, 0), 2)

    houve_cruzamento = False
    if res.boxes is not None and res.boxes.id is not None:
        caixas = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.int().tolist()

        for box, track_id in zip(caixas, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)

            if track_id in historico_y and track_id not in ids_processados:
                y_anterior = historico_y[track_id]

                if y_anterior < linha_y <= cy:
                    sentido_a += 1
                    ids_processados.add(track_id)
                    houve_cruzamento = True
                elif y_anterior > linha_y >= cy:
                    sentido_b += 1
                    ids_processados.add(track_id)
                    houve_cruzamento = True

            historico_y[track_id] = cy

    cv2.rectangle(annotated_frame, (10, 10), (360, 85), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Sentido A (Desce): {sentido_a}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Sentido B (Sobe): {sentido_b}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    out.write(annotated_frame)
    ultimo_frame = annotated_frame

    if houve_cruzamento:
        cv2.imwrite(caminho_print, annotated_frame)

    cv2.imshow("Pedagio Virtual Bidirecional", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if not os.path.exists(caminho_print) and ultimo_frame is not None:
    cv2.imwrite(caminho_print, ultimo_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo salvo em: {caminho_video}\nPrint salvo em: {caminho_print}")