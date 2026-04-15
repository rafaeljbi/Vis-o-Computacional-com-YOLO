from ultralytics import YOLO

# 1. Chamada do modelo pre-treinado
modelo = YOLO('yolo26n.pt')

# 2. Leitura do video (substitua o caminho do arquivo abaixo)
resultados = modelo(source=0, show=True, stream=True, conf=0.5)

# 3. Inferencia

for frame in resultados:
    # Retorna a probabilidade geral da cena
    print(frame.boxes.xyxy)

#IMPORTANTE: Para finalizar o video aperte a tecla "Q".