from ultralytics import YOLO
from datetime import datetime
import cv2
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

pasta_evidencias = os.path.join(OUTPUT_DIR, "evidencias")
os.makedirs(pasta_evidencias, exist_ok=True)
arquivo_csv = os.path.join(OUTPUT_DIR, "acessos.csv")

if not os.path.exists(arquivo_csv):
    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data_hora", "id_persistente", "caminho_imagem"])

video_source = os.path.join(BASE_DIR, "video.mp4")  # Ou 0 para webcam
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Erro ao abrir a fonte de vídeo.")
    exit()

largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

caminho_video = os.path.join(OUTPUT_DIR, "ex6_portaria.mp4")
caminho_print = os.path.join(OUTPUT_DIR, "ex6_print.jpg")
out = cv2.VideoWriter(caminho_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))

cv2.namedWindow("Auditoria de Portaria", cv2.WINDOW_NORMAL)

historico_y = {}
ids_registrados = set()
ultimo_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    linha_y = altura // 2

    results = model.track(source=frame, conf=0.5, classes=[0], persist=True, verbose=False)
    res = results[0]
    annotated_frame = res.plot()

    cv2.line(annotated_frame, (0, linha_y), (largura, linha_y), (0, 255, 255), 2)

    if res.boxes is not None and res.boxes.id is not None:
        caixas = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.int().tolist()

        for box, track_id in zip(caixas, ids):
            x1, y1, x2, y2 = map(int, box)
            cy = (y1 + y2) // 2

            if track_id in historico_y and track_id not in ids_registrados:
                y_anterior = historico_y[track_id]
                cruzou = (y_anterior < linha_y <= cy) or (y_anterior > linha_y >= cy)

                if cruzou:
                    ids_registrados.add(track_id)
                    agora = datetime.now()
                    timestamp_arquivo = agora.strftime("%Y%m%d_%H%M%S")
                    timestamp_log = agora.strftime("%Y-%m-%d %H:%M:%S")

                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(largura, x2), min(altura, y2)
                    crop = frame[y1_c:y2_c, x1_c:x2_c]

                    nome_img = f"pedestre_id{track_id}_{timestamp_arquivo}.jpg"
                    caminho_img = os.path.join(pasta_evidencias, nome_img)
                    caminho_relativo = f"evidencias/{nome_img}"

                    if crop.size > 0:
                        cv2.imwrite(caminho_img, crop)
                        cv2.imwrite(caminho_print, annotated_frame)

                        with open(arquivo_csv, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp_log, track_id, caminho_relativo])

                        print(f"[REGISTRADO] ID {track_id} salvo em resultados/{caminho_relativo}")

            historico_y[track_id] = cy

    out.write(annotated_frame)
    ultimo_frame = annotated_frame
    cv2.imshow("Auditoria de Portaria", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if not os.path.exists(caminho_print) and ultimo_frame is not None:
    cv2.imwrite(caminho_print, ultimo_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo salvo em: {caminho_video}\nPrint salvo em: {caminho_print}")