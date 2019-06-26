import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


# inicializa os dados e os rotulos
data = []
labels = []

for image_file in paths.list_images("extracted_letters"):
    # carregar a imagem e converte para escala de cinza
    img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY)
    # ajusta o tamanho da imagem para 20x20
    img = resize(img, 20, 20)
    # adiciona um terceira dimensão para a imagem para usar com Keras
    img = np.expand_dims(img, axis=2)
    # a label é o nome da pasta
    label = image_file.split(os.path.sep)[-2]
    # a imagem e a label são adicionadas para o conjunto de dados para treinamento
    data.append(img)
    labels.append(label)


# transforma em um array numpy para facilitar o uso dos dados
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# separa os dados entre conjunto de treino e de teste
(X_train, X_test, Y_train, Y_test) = train_test_split(
    data, labels, test_size=0.25, random_state=0
)
# converte as labels em one-hot encoding para o uso com Keras
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salva o mapeamento dos dados convertidos
with open("model_labels.dat", "wb") as f:
    pickle.dump(lb, f)

# constrói a rede neural
model = Sequential()
# primeira camada convolucional (max pooling)
model.add(
    Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu")
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# segunda camada convolucional (max pooling)
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# camada oculta com 500 nodos
model.add(Flatten())
model.add(Dense(500, activation="relu"))
# camada de saída com 32 nodos, um para cada caractere possível
model.add(Dense(32, activation="softmax"))
# o Keras constrói o modelo TensorFlow
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# treina a rede
model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    batch_size=32,
    epochs=10,
    verbose=1,
)
# salva o modelo
model.save(captcha_model.hdf5)
