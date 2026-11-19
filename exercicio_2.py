from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega o modelo e executa a predição na imagem
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
results = model.predict(source=os.path.join(BASE_DIR, "foto.jpg"), conf=0.5, verbose=False)

# Percorre as caixas detectadas manualmente (sem usar .plot())
boxes = results[0].boxes
linhas_relatorio = []

for i, box in enumerate(boxes, start=1):
    cls_id = int(box.cls[0])
    cls_nome = model.names[cls_id]
    confianca = float(box.conf[0]) * 100
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    linha = (
        f"Objeto {i}: Classe='{cls_nome}' (ID: {cls_id}) | "
        f"Confiança: {confianca:.2f}% | "
        f"Caixa: (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})"
    )
    print(linha)
    linhas_relatorio.append(linha)

# Salva automaticamente a saída do terminal em TXT dentro de resultados/
caminho_txt = os.path.join(OUTPUT_DIR, "ex2_metadados.txt")
with open(caminho_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(linhas_relatorio))

print(f"\nMetadados salvos automaticamente em: {caminho_txt}")