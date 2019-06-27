## Inteligência Artificial - UFFS 2019.1

#### Descrição:
Desenvolvimento de uma rede neural que resolve _Captchas_ simples.

#### Detalhes:
Trabalho desenvolvido utilizando linguagem [Python 3.5+](https://www.python.org/) e bibliotecas [OpenCV](https://github.com/skvark/opencv-python), [Keras](https://keras.io/) e [TensorFlow](https://www.tensorflow.org/).

### Instalação de dependências
```
pip install -r requirements.txt
```

### Execução
Depois de instaladas as dependências:

O tratamento dos _Captchas_ para o treinamento da rede:
```
python extract_single_letters_from_captchas.py
```

A criação do modelo/treino da rede:
```
python train_model.py
```

O uso do modelo para resolução de _Captchas_:
```
python solve_captchas_with_model.py
```
