# Exercicios YOLOv8 com OpenCV

Projeto em Python com sete exercicios de Visão Computacional usando o modelo YOLOv8 e OpenCV. Os exemplos abrangem deteccao em imagens, inspeccao de resultados, contagem em video, rastreamento, auditoria de passagens e deteccao de invasao em uma area definida.

**Autora:** Yasmin Cabrini Brancaleone

Este projeto foi desenvolvido para a materia de Visão Computacional.

## Recursos

- YOLOv8 pre-treinado com o arquivo `yolov8n.pt`
- Deteccao e rastreamento de objetos
- Processamento de imagens e videos
- Contadores e alertas em tempo real
- Registro de evidencias em imagem e arquivos CSV/TXT
- Saidas organizadas na pasta `resultados/`

## Estrutura do Projeto

```text
yolo_exercicios/
|-- exercicio_1.py
|-- exercicio_2.py
|-- exercicio_3.py
|-- exercicio_4.py
|-- exercicio_5.py
|-- exercicio_6.py
|-- exercicio_7.py
|-- foto.jpg
|-- video.mp4
|-- yolov8n.pt
|-- resultados/
|   |-- evidencias/
|   |-- acessos.csv
|   `-- ...
`-- Relatorio de Execucao.pdf
```

## Requisitos

- Python 3
- Ambiente virtual `venv`
- Windows, com PowerShell ou prompt de comando

Pacotes utilizados:

- `ultralytics`
- `opencv-python`

## Instalacao

No diretorio do projeto, crie o ambiente virtual:

```powershell
python -m venv venv
```

Ative o ambiente no Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Caso a execucao de scripts esteja bloqueada no PowerShell, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Instale as dependencias:

```powershell
pip install ultralytics opencv-python
```

## Execucao

Com o ambiente virtual ativado, execute cada exercicio separadamente:

```powershell
python exercicio_1.py
python exercicio_2.py
python exercicio_3.py
python exercicio_4.py
python exercicio_5.py
python exercicio_6.py
python exercicio_7.py
```

Os scripts utilizam os arquivos locais `foto.jpg`, `video.mp4` e `yolov8n.pt`. Nos scripts de video, tambem podem ser configurados para utilizar a webcam, alterando a fonte `video.mp4` para `0`.

As janelas de exibicao podem ser fechadas manualmente. Para encerrar os processamentos de video pelo teclado, pressione `q`.

## Descricao dos Exercicios

### 1. Classificador de Trafego Estatico

Arquivo: `exercicio_1.py`

Executa a deteccao em `foto.jpg`, filtrando exclusivamente veiculos do dataset COCO:

- carro: ID 2
- motocicleta: ID 3
- onibus: ID 5
- caminhao: ID 7

A imagem anotada e exibida em uma janela redimensionavel (`cv2.WINDOW_NORMAL`) e salva como `resultados/ex1_trafego_detectado.jpg`.

### 2. Inspetor de Metadados e Bounding Boxes

Arquivo: `exercicio_2.py`

Percorre manualmente as caixas presentes em `results[0].boxes`, sem utilizar `results.plot()`, e imprime no terminal:

- nome da classe
- ID da classe
- porcentagem de confianca
- coordenadas `(x1, y1, x2, y2)`

Os dados sao salvos em `resultados/ex2_metadados.txt`.

### 3. Filtro de Confianca com Alerta

Arquivo: `exercicio_3.py`

Aplica um limiar de confianca de 70% ao processar `foto.jpg`. Quando uma pessoa com ID 0 e confianca superior a 85% e encontrada, exibe um alerta no terminal.

A imagem processada e salva como `resultados/ex3_filtro_confianca.jpg`.

### 4. Contador em Tempo Real com Resumo Dinamico

Arquivo: `exercicio_4.py`

Processa `video.mp4` e conta pessoas e carros em cada quadro por meio de `collections.Counter`. O placar dinâmico é exibido no canto superior esquerdo com `cv2.putText`.

Gerados:

- `resultados/ex4_contador.mp4`
- `resultados/ex4_print.jpg`

### 5. Linha de Pedagio Virtual Bidirecional

Arquivo: `exercicio_5.py`

Utiliza rastreamento persistente com `model.track(persist=True)` e uma linha horizontal central para contar veiculos que cruzam em cada sentido:

- Sentido A: de cima para baixo
- Sentido B: de baixo para cima

Os IDs rastreados evitam contagens duplicadas do mesmo objeto.

Gerados:

- `resultados/ex5_pedagio.mp4`
- `resultados/ex5_print.jpg`

### 6. Auditoria com Log em CSV e Captura de Evidencias

Arquivo: `exercicio_6.py`

Monitora pedestres atravessando uma linha virtual. Para cada pessoa nova que cruza a linha:

- cria um recorte individual da pessoa;
- salva o recorte em `resultados/evidencias/`;
- registra data/hora, ID persistente e caminho da imagem em `resultados/acessos.csv`.

Gerados:

- `resultados/ex6_portaria.mp4`
- `resultados/ex6_print.jpg`
- arquivos de evidencias em `resultados/evidencias/`

### 7. Detector de Invasao de Area Proibida

Arquivo: `exercicio_7.py`

Define uma região poligonal (ROI) proporcional ao tamanho do quadro e verifica o centro de cada deteccao de pessoa com `cv2.pointPolygonTest`.

Quando uma pessoa entra no poligono:

- a borda da area fica vermelha;
- um alerta visual de invasao é exibido;
- um print da ocorrencia pode ser salvo.

Gerados:

- `resultados/ex7_invasao.mp4`
- `resultados/ex7_print_invasao.jpg`

## Saidas

A pasta `resultados/` ja contem os resultados produzidos pelos exercicios:

| Arquivo | Descricao |
|---|---|
| `ex1_trafego_detectado.jpg` | Trafego detectado na imagem estatica |
| `ex2_metadados.txt` | Metadados das deteccoes |
| `ex3_filtro_confianca.jpg` | Resultado filtrado por confianca |
| `ex4_contador.mp4` | Video com contagem de pessoas e carros |
| `ex4_print.jpg` | Print do contador |
| `ex5_pedagio.mp4` | Video do pedagio virtual |
| `ex5_print.jpg` | Print de cruzamento no pedagio |
| `ex6_portaria.mp4` | Video da auditoria de pedestres |
| `ex6_print.jpg` | Print da deteccao registrada |
| `ex7_invasao.mp4` | Video da monitoracao da area proibida |
| `ex7_print_invasao.jpg` | Evidencia visual da invasao |
| `acessos.csv` | Log de pessoas que cruzaram a linha |
| `evidencias/` | Recortes individuais dos pedestres |

## Observacoes

- O modelo `yolov8n.pt` deve permanecer na raiz do projeto.
- O desempenho pode variar conforme a resolução do video e o hardware disponível.
- Reinicie o terminal antes de alterar o ambiente virtual, caso o Python ativo não corresponda ao criado em `venv`.
- Este projeto tem finalidade educacional
