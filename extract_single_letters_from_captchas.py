import os
import os.path
import cv2
import glob
import imutils

# Armazanas os arquivos de imagens dos captchas
captcha_files = glob.glob("captchas_for_trainning/*")
number_of_files = len(captcha_files)
counts = {}

print("Processando imagem {}/".format(number_of_files), end="")
for i, captcha_file in enumerate(captcha_files):
    if i > 0: print('\b'*4, end="")
    print('{:4d}'.format(i + 1), end="")

    # Lê o nome dos arquivos sem o caminho e sem a extensão
    filename = os.path.splitext(os.path.basename(captcha_file))[0]
    # Converte a imagem para uma imagem em escala de cinza
    grayscale = cv2.cvtColor(cv2.imread(captcha_file), cv2.COLOR_BGR2GRAY)
    # Gerar borda ao redor dos caracteres
    grayscale = cv2.copyMakeBorder(grayscale, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    # Usamos um limiar para converter a imagem em um imagem em preto e branco
    thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Encontra o contorno dos caracteres
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Lista de regiões dos caracteres encontrados na imagens
    letter_image_regions = []

    # Lista os 4 contornos dentro da imagem
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1.25:
            # Quando o contorno é mais largo provavelmente são duas letras
            # Então o contorno é dividido no meio
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    if len(letter_image_regions) != 4:
        continue

    # Ordena os caracteres de acordo com a posição na imagem
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    for letter_bounding_box, letter_text in zip(letter_image_regions, filename):
        # Pega as coordenadas da imagem
        x, y, w, h = letter_bounding_box
        # Extrai o caracter da imagem com uma margem de dois pixels
        letter_image = grayscale[y - 2:y + h + 2, x - 2:x + w + 2]
        # Pega o caminho da pasta onde será salvo o caracter
        path = os.path.join("extracted_letters", letter_text)
        # cria o diretorio caso não exista
        if not os.path.exists(path):
            os.makedirs(path)

        # Salva o contador para o nome da imagem
        count = counts.get(letter_text, 1)
        p = os.path.join(path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # Incrementa o contador do caracter especifico
        counts[letter_text] = count + 1

print("")
