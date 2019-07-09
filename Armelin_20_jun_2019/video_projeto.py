import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

####################################### FUNÇÕES ######################################

def grayscale(img):                                                      # Função para deixar a imagem em escalas de cinza
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img):                                                  # Função para aplicar filtragem gaussiana na imagem (borrar levemente)
	return cv2.GaussianBlur(img, (5, 5), 0)                              # Ela ajuda a diminuir o ruído de cores da mesma

def canny(img, low_threshold, high_threshold):                           # Função para a detecção de bordas na imagem
	return cv2.Canny(img, low_threshold, high_threshold, apertureSize=3)

def plotar(img, **kwargs):                                               # O '**kwargs' cria um dicionário com os parâmetros, onde a variável é a KEY e o valor é o VALUE
	mascara = kwargs.get('mascara', None)                                # Retorna o VALUE da KEY 'parâmetro', caso a KEY não exista, retorna 'None'
	if mascara == None: plt.imshow(img)                                  # Se não for passado parâmetro, apenas usa a imagem
	elif mascara == 'gray': plt.imshow(img, cmap='gray')                 # Se o parâmetro for 'gray', usa a imagem com um mapeamento em escalas de cinza
	else: plt.imshow(cv2.cvtColor(img, mascara))                         # Se for usar alguma conversão com 'cvtColor', usa esta opção
	plt.show()                                                           # Plota a imagem

def GammaAdjust(img, gamma=0.8):                                         # Função para o ajuste de gama, seu intuito é
	invGamma = 1.0 / gamma                                               # o de equalizar razoavelmente os claros com
	table = np.array([((i / 255.0) ** invGamma) * 255                    # escuros, reduzindo o ruído dos mesmos
		for i in np.arange(0, 256)]).astype('uint8')
	return cv2.LUT(img, table)

def BinaryFilter(img):                                                   # Função para binarizar a imagem, ou seja,
	_, threshold = cv2.threshold(img, 100, 160, cv2.THRESH_BINARY)       # as cores se tornarão OU totalmente branco
	return threshold                                                     # OU totalmente preto (255 ou 0)

################################## CÓDIGO PRINCIPAL ##################################

vid = cv2.VideoCapture('video_2.mp4')                                                         # Lê o vídeo

while(vid.isOpened()):                                                                        # Mantém o loop enquanto não for pressionado a tecla 'q'
	ret, frame = vid.read()                                                                   # Lê o frame atual

	frame = GammaAdjust(frame)                                                                # Ajusta os realces da imagem (aproxima os claros dos escuros e vice-versa)

	vertices = np.array([[50,720], [550, 450], [600,450], [1280, 600], [1280, 720],[50,720]]) # Cria uma lista que contém os vértices do polígono
	mask = np.zeros_like(frame)                                                               # Cria uma imagem em preto com a mesma dimensão da imagem original

	# Esta parte serve para preparar para deixar a parte interna do polígono totalment branca, pois quando usarmos a função
	# 'bitwise_and', ele fará a seguinte classificação: ambas as imagens serão avaliadas simultâneamente 'pixel a pixel', e
	# quando os pixels de ambas as imagens forem diferentes de zero (não pretos), o pixel resultante será a cor não-branca
	# (pense na cor branca como o valor 1 e outras cores como 0,89; 0,50; etc. e a função multiplica esses valores, gerando o
	# valor diferente de 1), resultando em uma imagem na forma do polígono selecionado, e o resto dela estando preta.
	if len(frame.shape) > 2:                                                               # Vê se a imagem tem canais de cores (altura, comprimento, canais de cores)
	    channel_count = frame.shape[2]                                                     # Coloca a quantidade de canais de cores na variável 'channel_count'
	    ignore_mask_color = (255,) * channel_count                                         # Cria a tupla com a quantidade de brancos igual aos canais de cores
	else:
	    ignore_mask_color = 255                                                            # Caso não possua canais de cores, deixa a imagem branca também

	cv2.fillConvexPoly(mask, vertices, ignore_mask_color)                                  # Cria um polígono branco na imagem totalmente preta
	masked_image = cv2.bitwise_and(frame, mask)                                            # Concatena a imagem 'img' com a máscara

	gray = grayscale(masked_image)                                                         # Deixa a imagem em escalas de cinza

	blur_gray = gaussian_blur(gray)                                                        # Aplica desfoque gaussiano

	binario = BinaryFilter(blur_gray)                                                      # Binariza a imagem (só preto ou branco)

	edges = canny(binario, 30, 230)                                                        # Detecta as bordas na imagem

	edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)                                        # Coloca canais de cores na imagem 'edges', para converter as linhas brancas em vermelhas
	edges[np.where((edges == [255,255,255]).all(axis = 2))] = [0,0,255]                    # Substitui as cores brancas por vermelhas

	img_fim = cv2.bitwise_or(frame, edges)                                                 # Coloca as linhas vermelhas na imagem final

	cv2.imshow('frame',img_fim)                                                            # Mostra o frame atual
	if cv2.waitKey(1) & 0xFF == ord('q'):                                                  # Se a tecla 'q' for pressionada, paramos de ler as imagens
	    break

cv2.destroyAllWindows()                                                                    # Fecha todas as janelas