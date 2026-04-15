from ultralytics import YOLO

# 1. Chamada do modelo pre-treinado
modelo = YOLO('yolo26n-cls.pt')

# 2. Leitura do video (substitua o caminho do arquivo abaixo)
resultados = modelo(source=r'D:\Tese_UPM\Capítulo_Livro\Videos\video_exemplo.mp4', show=True, stream=True)

# 3. Inferencia

for frame in resultados:
    # Retorna a probabilidade geral da cena
    print(frame.probs.top1)

#IMPORTANTE: Para finalizar o video aperte a tecla "Q".