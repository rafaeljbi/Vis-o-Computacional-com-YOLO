# IMPORTANTE: Instale a biblioteca ultralytics antes de rodar o código

# Execute no terminal:
# pip install ultralytics
from ultralytics import YOLO

# 1. Chamada do modelo pre-treinado
modelo = YOLO('yolo26n.pt')

# 2. Leitura do video (substitua o caminho do arquivo abaixo)
resultados = modelo(source=r'D:\Tese_UPM\Capítulo_Livro\Videos\video_exemplo2.mp4', show=True, stream=True)

# 3. Inferencia

for frame in resultados:
    # Retorna a probabilidade geral da cena
    print(frame.boxes.xyxy)

#IMPORTANTE: Para finalizar o video aperte a tecla "Q".
