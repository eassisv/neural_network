from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "captchas_for_validation"

# Carrega o mapeamento das labels Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Carrega o modelo da rede
model = load_model(MODEL_FILENAME)

# Pega Captchas aleatórios da pasta de imagens para validação
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

# Loop para testar todas as imagens
for image_file in captcha_image_files:
    # Carrega a imagens e muda ela para escala de cinza
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adiciona preenchimento nas bordas
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # Usa limiar para converter a imagem para preto e branco
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Encontra os contornos
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0]

    letter_image_regions = []

    # Extrai o caractere de cada contorno
    for contour in contours:
        # Pega as coordenadas do contorno
        (x, y, w, h) = cv2.boundingRect(contour)

        # Verifica se há sobreposição de caracteres e divide o contorno em dois caso haja
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # Caso não sejam quatro caracteres a imagem é ignorada
    if len(letter_image_regions) != 4:
        continue

    # Ordena baseado na coordenada x para garantir que a imagem é processada da esquerda para a direita
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Cria a imagem de saída
    output = cv2.merge([image] * 3)
    predictions = []

    # Tratamento dos caracteres encontrados
    for letter_bounding_box in letter_image_regions:
        # Pega as coordenadas do caractere
        x, y, w, h = letter_bounding_box

        # Extrai o caractere e adiciona uma margem de 2px
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Ajusta a imagem para 20x20px
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Converte a imagem para uma lista numpy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Faz as predições
        prediction = model.predict(letter_image)

        # Converte com base no mapeamento das labels
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # Desenha os resultados na imagem
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Exibe o resultado
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    # Exibe a imagem com o resultado
    cv2.imshow("Output", output)
    cv2.waitKey()
