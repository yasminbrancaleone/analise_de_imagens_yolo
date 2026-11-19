from ultralytics import YOLO
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega o modelo e executa a predição com limiar de confiança 0.70
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
results = model.predict(source=os.path.join(BASE_DIR, "foto.jpg"), conf=0.7, verbose=False)

# Verifica se alguma pessoa (ID 0) tem confiança superior a 85%
alerta_disparado = False
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    confianca = float(box.conf[0])

    if cls_id == 0 and confianca > 0.85:
        alerta_disparado = True
        break

mensagem = (
    "ALERTA: Indivíduo identificado com alta certeza!"
    if alerta_disparado
    else "Nenhuma pessoa identificada com confiança superior a 85%."
)
print(mensagem)

# Salva automaticamente a imagem anotada e o log de alerta em resultados/
annotated_frame = results[0].plot()
caminho_img = os.path.join(OUTPUT_DIR, "ex3_filtro_confianca.jpg")

cv2.imwrite(caminho_img, annotated_frame)
with open(caminho_txt, "w", encoding="utf-8") as f:
    f.write(mensagem + "\n")

print(f"Salvo automaticamente em: {caminho_img} e {caminho_txt}")

# 4. Exibe a imagem processada
cv2.namedWindow("Filtro de Confianca", cv2.WINDOW_NORMAL)
cv2.imshow("Filtro de Confianca", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()