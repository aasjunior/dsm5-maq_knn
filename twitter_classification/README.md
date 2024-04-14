# Análise de sentimento em tweets

<p align='justify'>Base de dados para análise de sentimento em tweets multilíngues</p>

**Base URL:** &nbsp;[Dataset Kaggle](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)

**Quantidade de atributos:** 2

**Tipos de dados dos atributos:** Mensagens de texto (tweet) 

**Instâncias:** 27481

**Classes:**
* Positivo
* Negativo
* Neutro

## Algoritmo KNN

### Classe DataModel

<p align='justify'>A classe <code>DataModel</code> é usada para carregar e normalizar um conjunto de dados a partir de um arquivo CSV. Ela tem três funções principais:</p>

- `__init__(self, file_path, numeric_cols, categorical_cols)`: Este é o construtor da classe. Ele é chamado quando um objeto DataModel é criado e recebe o caminho do arquivo, as colunas numéricas e as colunas categóricas como argumentos.

- `load_data(self)`: Esta função tenta ler o arquivo CSV especificado no construtor. Se ocorrer um erro durante a leitura do arquivo, ela lança uma exceção.

- `normalize_data(self)`: Esta função normaliza os dados removendo linhas duplicadas e com valores ausentes. Se ocorrer um erro durante a normalização, ela lança uma exceção.

- `load_by_line(self)`: Esta função lê o arquivo CSV linha por linha quando ocorre um erro de análise durante a leitura normal do arquivo. Ela retorna um DataFrame Pandas com os dados lidos.

- `print_data_info(data, label_column)`: Esta função estática imprime informações sobre o conjunto de dados, incluindo o número de instâncias, o número de atributos, os nomes dos atributos, o número de linhas duplicadas, o número de linhas com dados faltantes e as classes na coluna de rótulos.

- `remove_extra_characters(data_column, character)`: Esta função estática remove caracteres extras de uma coluna de dados. Ela recebe uma coluna de dados e um caractere como argumentos e retorna a coluna de dados com o caractere removido.

<br>

```Python
import pandas as pd
import csv
from traceback import format_exception

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
        
        except pd.errors.ParserError as e:
            try:
                return self.load_by_line()
            
            except Exception as e:
                raise Exception(f'Erro ao analisar o arquivo {self.file_path}.\n\nTraceback: {e}')
        
        except Exception as e:
            raise Exception(f'Erro ao tentar ler o arquivo: {e}')
    

    def normalize_data(self):
        try:
            data = self.load_data()
            self.print_data_info(data, data.iloc[:, 3])
            
            data.dropna(inplace=True)
            data = data.drop_duplicates()

            # duplicated = data.duplicated().sum()

            return data
        
        except Exception as e:
            raise Exception(f'Erro na normalização: {e}')
        

    def load_by_line(self):
        error_count = 0
        data_list = []

        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                try:
                    if len(row) != len(headers):
                        raise ValueError('Número de campos na linha não corresponde ao número de campos no cabeçalho')
                    
                    data_list.append(row)
                    
                except Exception as e:
                    error_count += 1
                    continue
        
        print(f'Carregamento do arquivo {self.file_path}:\nNúmero de linhas com erro: {error_count}\n\n')
    
        return pd.DataFrame(data_list, columns=headers)


    @staticmethod
    def print_data_info(data, label_column):
        print(f'Número de instâncias: {data.shape[0]}')
        print(f'Número de atributos: {data.shape[1]}')
        print(f'Nome dos atributos: {data.columns.tolist()}')
        print(f'Número de linhas duplicadas: {data.duplicated().sum()}')
        print(f'Número de linhas com dados faltantes: {data.isnull().sum().sum()}')

        if label_column.dtype == 'object':
            print(f'\nClasses na coluna {label_column.name}:')
            print(label_column.value_counts())
            print('\n')


    @staticmethod
    def remove_extra_characters(data_column, character):
        return data_column.str.replace(character, '')
```

### Normalização dos dados

<p align='justify'>Para normalização dos dados, além do método <code>normalize_data</code> da classe <code>DataModel</code>, também foi criado o serviço de normalização para atender a base de dados específica:</p>

- `normalize_data(model)`: Esta função recebe o modelo de dados como entrada e realiza várias operações para normalizar os dados:
    - Aplica a função `preprocess_message` a cada mensagem no modelo de dados.
    - Transforma as mensagens em vetores TF-IDF usando o `TfidfVectorizer` do scikit-learn.
    - Normaliza os vetores TF-IDF para o intervalo [0, 1] usando o `MinMaxScaler` do scikit-learn.
    - Codifica os rótulos da categoria usando a função `encode_labels`.
    - Retorna os dados normalizados e os rótulos codificados.

```Python
def normalize_data(model):
    try:
        data = preprocess_data(model)
        data['text'] = data['text'].apply(preprocess_message)
        data['selected_text'] = data['selected_text'].apply(preprocess_message)

        messages = vectorizer_data(data)

        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(messages)

        labels = encode_labels(data.iloc[:, 2])

        return data_normalized, labels
    
    except Exception as e:
        raise Exception(f'\nErro na normalização dos dados: {e}')

```
<br>

- `preprocess_data(data)`: Esta função recebe os dados como entrada, remove a coluna ‘textID’, remove caracteres extras das colunas e dos rótulos, balanceia os dados usando a função balanced_data, imprime informações sobre os dados após o pré-processamento e retorna os dados pré-processados.

```Python
def preprocess_data(data):
    data = data.drop('textID', axis=1)
    data.columns = DataModel.remove_extra_characters(data.columns, ';')
    data.iloc[:, 2] = DataModel.remove_extra_characters(data.iloc[:, 2], ';')
    data.iloc[:, 2] = DataModel.remove_extra_characters(data.iloc[:, 2], '"')
    
    data = balanced_data(data, 2000)

    print('Dados depois do pré-processamento:')
    DataModel.print_data_info(data, data.iloc[:, 2])

    return data

```
<br>

- `balanced_data(data, max_instances)`: Esta função recebe os dados e o número máximo de instâncias como entrada, divide os dados por classe, seleciona um número de instâncias de cada classe igual ao menor valor entre o número de instâncias disponíveis e max_instances, e retorna a concatenação dos dados selecionados de cada classe.

```Python
def balanced_data(data, max_instances):
    try:
        # Dividir os dados por classe
        data_negative = data[data['sentiment'] == 'negative']
        data_neutral = data[data['sentiment'] == 'neutral']
        data_positive = data[data['sentiment'] == 'positive']

        # Selecionar um número igual de instâncias de cada classe
        data_negative = data_negative.sample(min(len(data_negative), max_instances))
        data_neutral = data_neutral.sample(min(len(data_neutral), max_instances))
        data_positive = data_positive.sample(min(len(data_positive), max_instances))

        # Concatenar os resultados
        return pd.concat([data_negative, data_neutral, data_positive])
    
    except Exception as e:
        raise Exception(f'\nErro ao tentar balancear os dados: {e}')

```
<br>

- `preprocess_message(message)`: Esta função recebe uma mensagem como entrada e realiza duas operações de pré-processamento:
    - Remove caracteres não alfabéticos da mensagem usando uma expressão regular.
    - Converte a mensagem para letras minúsculas.
    - Retorna a mensagem pré-processada.

```Python
def preprocess_message(message):
    try:
        message = re.sub('[^a-zA-Z]', ' ', message)
        message = message.lower()
        
        return message
    
    except Exception as e:
        raise Exception(f'\nErro no pré-processamento dos dados: {e}')

```
<br>

- `vectorizer_data(data)`: Esta função recebe os dados como entrada, aplica o TfidfVectorizer do scikit-learn aos textos e aos textos selecionados, e retorna a concatenação dos dois conjuntos de vetores.


```Python

def vectorizer_data(data):
    try:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(pd.concat([data['text'], data['selected_text']]))

        text = vectorizer.transform(data['text'])
        selected_text = vectorizer.transform(data['selected_text'])

        return np.concatenate((text.toarray(), selected_text.toarray()), axis=1)
    
    except Exception as e:
        raise Exception(f'\nErro na vetorização dos dados de entrada: {e}')

```
<br>

- `encode_labels(model)`: Esta função recebe o modelo de dados como entrada e codifica os rótulos da categoria usando o `LabelEncoder` do scikit-learn. Retorna os rótulos codificados.

```Python
def encode_labels(label_column):
    try:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(label_column)
       
        return labels

    except Exception as e:
        raise Exception(f'\nErro na conversão dos rótulos: {e}')
```
<br>

### Variação do número de vizinhos próximos (k)

<p align='justify'>Para análise paramétrica do algoritmo KNN, é utilizada a função <code>variable_k</code>:</p>

1. Recebe os dados e um dicionário de valores para o algoritmo KNN como argumentos.
2. Normaliza os dados usando a função `normalize_data`, retorna os atributos (X) e as classes (Y)
3. Para cada valor de `k` no dicionário, treina um modelo KNN e avalia sua acurácia.
4. Imprime a acurácia do modelo para cada valor de `k`.
5. Se ocorrer um erro em qualquer parte do processo, lança uma exceção.

<br>

```Python
from model.KNNModel import KNNModel
from .normalize import normalize_data

def variable_k(data, knn_values):
    try:
        X, Y = normalize_data(data)
        
        for k in knn_values['neighboors']:
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

    numeric_cols = ['text', 'selected_text']
    categorical_cols = ['sentiment']

    model = DataModel('db/tweets.da"ta', numeric_cols, categorical_cols)

    variable_k(model.data, knn_values)

except Exception as e:
    print(f'Ocorreu um erro:\n{e}')
```

**Saídas:**

```Terminal
Dados depois do pré-processamento:
Número de instâncias: 6000
Número de atributos: 3
Nome dos atributos: ['text', 'selected_text', 'sentiment']
Número de linhas duplicadas: 0
Número de linhas com dados faltantes: 0

Classes na coluna sentiment:
sentiment
negative    2000
neutral     2000
positive    2000
Name: count, dtype: int64

# 1ª
Para k=3, a acurácia foi de 53.78%
Para k=5, a acurácia foi de 57.22%
Para k=7, a acurácia foi de 61.50%

# 2ª
Para k=3, a acurácia foi de 55.61%
Para k=5, a acurácia foi de 55.22%
Para k=7, a acurácia foi de 57.72%

# 3ª
Para k=3, a acurácia foi de 57.11%
Para k=5, a acurácia foi de 55.78%
Para k=7, a acurácia foi de 54.89%

# 4ª
Para k=3, a acurácia foi de 60.56%
Para k=5, a acurácia foi de 60.33%
Para k=7, a acurácia foi de 55.89%

# 5ª
Para k=3, a acurácia foi de 46.11%
Para k=5, a acurácia foi de 47.78%
Para k=7, a acurácia foi de 59.56%
```