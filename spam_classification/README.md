# Classificação de mensagens de e-mail (Spam)
<p align='justify'>Base de dados para classificação de mensagens de texto como spam</p>

**Base URL:** &nbsp;[Dataset Kaggle](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification/data)

**Quantidade de atributos:** 1

**Tipos de dados dos atributos:** Mensagens de texto (e-mail) 

**Instâncias:** 5574

**Classes:**

* Spam
* Ham (normal)
 

## Algoritmo KNN

### Normalização dos dados

<p align='justify'>Para normalização dos dados, além do método <code>normalize_data</code> da classe <code>DataModel</code>, também foi criado o serviço de normalização para atender a base de dados específica:</p>

- `preprocess_message(self, message)`: Esta função recebe uma mensagem como entrada e realiza duas operações de pré-processamento:
    - Remove caracteres não alfabéticos da mensagem usando uma expressão regular.
    - Converte a mensagem para letras minúsculas.
    - Retorna a mensagem pré-processada.

- `encode_labels(self, model)`: Esta função recebe o modelo de dados como entrada e codifica os rótulos da categoria usando o `LabelEncoder` do scikit-learn. Retorna os rótulos codificados.

- `normalize_data(self)`: Esta função recebe o modelo de dados como entrada e realiza várias operações para normalizar os dados:
    - Aplica a função `preprocess_message` a cada mensagem no modelo de dados.
    - Transforma as mensagens em vetores TF-IDF usando o `TfidfVectorizer` do scikit-learn.
    - Normaliza os vetores TF-IDF para o intervalo [0, 1] usando o `MinMaxScaler` do scikit-learn.
    - Codifica os rótulos da categoria usando a função `encode_labels`.
    - Retorna os dados normalizados e os rótulos codificados.

<br>

```Python
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def normalize_data(model):
    try:
        model['Message'] = model['Message'].apply(preprocess_message)

        vectorizer = TfidfVectorizer()
        messages = vectorizer.fit_transform(model['Message'])

        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(messages.toarray())

        labels = encode_labels(model)

        return data_normalized, labels
    
    except Exception as e:
        raise Exception(f'Erro na normalização dos dados: {e}')


def encode_labels(model):
    try:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(model['Category'])
    
        return labels

    except Exception as e:
        raise Exception(f'Erro na conversão dos rótulos: {e}')


def preprocess_message(message):
    try:
        # Remove non-alphabetic characters
        message = re.sub('[^a-zA-Z]', ' ', message)
        
        # Convert to lower case
        message = message.lower()
        
        return message
    
    except Exception as e:
        raise Exception(f'Erro no pré-processamento dos dados: {e}')
```

### Variação do número de vizinhos próximos (k)

<p align='justify'>Para análise paramétrica do algoritmo KNN, é utilizada a função <code>variable_k</code>:</p>

1. Recebe os dados e um dicionário de valores para o algoritmo KNN como argumentos.
2. Para cada valor de `k` no dicionário, normaliza os dados, treina um modelo KNN e avalia sua acurácia.
3. Imprime a acurácia do modelo para cada valor de `k`.
4. Se ocorrer um erro em qualquer parte do processo, lança uma exceção.

<br>

```Python
from model.KNNModel import KNNModel
from .normalize import normalize_data

def variable_k(data, knn_values):
    try:
        for k in knn_values['neighboors']:
            X, Y = normalize_data(data)

            knn = KNNModel(data, k)
            accuracy = knn.train_and_evaluate(X, Y, knn_values['test_size'], knn_values['train_size'])

            print(f'Para k={k}, a acurácia foi de {accuracy}')
        
    except Exception as e:
        raise Exception(f'Erro na execução de variação de vizinhos: {e}')
```

### Execução e avaliação

<p align='justify'>Durante a execução, para as diferentes quantidades de vizinhos (k), foram apresentados diferentes valores de acurácia. Embora na maioria das execuções tenha indicado que o menor número de vizinhos resulta em um valor maior na acurácia, não foi unanime em todos os casos.</p>

```Python
from model.DataModel import DataModel
from service.variable import variable_k

try:
    knn_values = {
        'neighboors': [3, 5, 7],
        'test_size': 0.3,
        'train_size': 0.7
    }

    numeric_cols = ['Message']
    categorical_cols = ['Category']

    model = DataModel('db/spam_text_message.data', numeric_cols, categorical_cols)

    variable_k(model.data, knn_values)

except Exception as e:
    print(f'Ocorreu um erro:\n{e}')

```

**Saídas:**

```Terminal
# 1ª)
Para k=3, a acurácia foi de 91.41%
Para k=5, a acurácia foi de 91.09%
Para k=7, a acurácia foi de 89.99%

# 2ª)
Para k=3, a acurácia foi de 89.92%
Para k=5, a acurácia foi de 91.09%
Para k=7, a acurácia foi de 89.08%

# 3ª)
Para k=3, a acurácia foi de 92.51%
Para k=5, a acurácia foi de 90.70%
Para k=7, a acurácia foi de 88.70%

# 4ª)
Para k=3, a acurácia foi de 92.89%
Para k=5, a acurácia foi de 90.25%
Para k=7, a acurácia foi de 89.41%
```