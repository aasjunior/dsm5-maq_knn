# Implementação do algoritmo KNN

Modelos:

1. &nbsp;[Spam Classification](spam_classification/README.md)
2. &nbsp;[Twitter Classification](twitter_classification/README.md)

## Algoritmo KNN

### Classe DataModel

<p align='justify'>A classe <code>DataModel</code> é usada para carregar e normalizar um conjunto de dados a partir de um arquivo CSV. Ela tem três funções principais:</p>

- `__init__(self, file_path, numeric_cols, categorical_cols)`: Este é o construtor da classe. Ele é chamado quando um objeto `DataModel` é criado e recebe o caminho do arquivo, as colunas numéricas e as colunas categóricas como argumentos.

- `load_data(self)`: Esta função tenta ler o arquivo CSV especificado no construtor. Se ocorrer um erro durante a leitura do arquivo, ela lança uma exceção.

- `normalize_data(self)`: Esta função normaliza os dados removendo linhas duplicadas e com valores ausentes. Se ocorrer um erro durante a normalização, ela lança uma exceção.

<br>

```Python
import pandas as pd

class DataModel:
    def __init__(self, file_path, numeric_cols, categorical_cols):
        self.file_path = file_path
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.data = self.normalize_data()


    def load_data(self):
        try:
            return pd.read_csv(self.file_path)
        
        except FileNotFoundError:
            raise FileNotFoundError(f'O arquivo {self.file_path} não foi encontrado.')
        
        except pd.errors.ParserError:
            raise Exception(f'Erro ao analisar o arquivo {self.file_path}.')
        
        except Exception as e:
            raise Exception(f'Erro ao tentar ler o arquivo: {e}')
    

    def normalize_data(self):
        try:
            data = self.load_data()
            data.dropna(inplace=True)
            data = data.drop_duplicates()
            # duplicated = data.duplicated().sum()

            return data
        
        except Exception as e:
            raise Exception(f'Erro na normalização: {e}')
``` 

### Classe KNNModel

<p align='justify'>A classe <code>KNNModel</code> é usada para treinar e avaliar um modelo de classificação K-Nearest Neighbors (KNN). Ela tem quatro funções principais:</p>

- `__init__(self, data_model, neighboor)`: Este é o construtor da classe. Ele é chamado quando um objeto `KNNModel` é criado e recebe o modelo de dados e o número de vizinhos como argumentos.

- `train(self, X_train, Y_train)`: Esta função treina o modelo KNN usando os dados de treinamento fornecidos.

- `predict(self, X_test)`: Esta função usa o modelo KNN treinado para fazer previsões nos dados de teste fornecidos.

- `evaluate(self, Y_test, predictions)`: Esta função avalia a acurácia do modelo KNN comparando as previsões do modelo com os verdadeiros rótulos dos dados de teste.

- `train_and_evaluate(self, X, Y, test_size=0.3, train_size=0.7)`: Esta função divide os dados em conjuntos de treinamento e teste, treina o modelo nos dados de treinamento, faz previsões nos dados de teste e avalia a acurácia do modelo.

<br>

```Python
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNNModel:
    def __init__(self, data_model, neighboor):
        self.data_model = data_model
        self.neighboor = neighboor
        self.knn = KNeighborsClassifier(self.neighboor)

    def train(self, X_train, Y_train):
        try:
            self.knn.fit(X_train, Y_train)
        
        except Exception as e:
            raise Exception(f'Erro no treinamento: {e}')

    def predict(self, X_test):
        try:
            return self.knn.predict(X_test)
     
        except Exception as e:
            raise Exception(f'Erro na previsão: {e}')


    def evaluate(self, Y_test, predictions):
        try:
            accuracy = accuracy_score(Y_test, predictions) * 100
            return "%.2f%%" % accuracy
        
        except Exception as e:
            raise Exception(f'Erro durante a avaliação: {e}')


    def train_and_evaluate(self, X, Y, test_size=0.3, train_size=0.7):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

        self.train(X_train, Y_train)

        predictions = self.predict(X_test)

        return self.evaluate(Y_test, predictions)
```


