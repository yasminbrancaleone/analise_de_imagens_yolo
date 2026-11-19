from ultralytics import YOLO
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega o modelo pré-treinado
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

# Executa a predição filtrando apenas veículos (car, motorcycle, bus, truck)
results = model.predict(source=os.path.join(BASE_DIR, "foto.jpg"), conf=0.5, classes=[2, 3, 5, 7])

# Gera a imagem anotada e salva na pasta resultados/
annotated_frame = results[0].plot()
caminho_saida = os.path.join(OUTPUT_DIR, "ex1_trafego_detectado.jpg")
cv2.imwrite(caminho_saida, annotated_frame)
print(f"Salvo automaticamente em: {caminho_saida}")

# Exibe em janela redimensionável
cv2.namedWindow("Classificador de Trafego", cv2.WINDOW_NORMAL)
cv2.imshow("Classificador de Trafego", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()