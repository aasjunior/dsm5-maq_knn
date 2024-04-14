from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from model.DataModel import DataModel
import numpy as np
import pandas as pd
import re

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


def vectorizer_data(data):
    try:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(pd.concat([data['text'], data['selected_text']]))

        text = vectorizer.transform(data['text'])
        selected_text = vectorizer.transform(data['selected_text'])

        return np.concatenate((text.toarray(), selected_text.toarray()), axis=1)
    
    except Exception as e:
        raise Exception(f'\nErro na vetorização dos dados de entrada: {e}')


def encode_labels(label_column):
    try:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(label_column)
       
        return labels

    except Exception as e:
        raise Exception(f'\nErro na conversão dos rótulos: {e}')


def preprocess_message(message):
    try:
        # Remove non-alphabetic characters
        message = re.sub('[^a-zA-Z]', ' ', message)
        
        # Convert to lower case
        message = message.lower()
        
        return message
    
    except Exception as e:
        raise Exception(f'\nErro no pré-processamento dos dados: {e}')


def preprocess_data(data):
    data = data.drop('textID', axis=1)
    data.columns = DataModel.remove_extra_characters(data.columns, ';')
    data.iloc[:, 2] = DataModel.remove_extra_characters(data.iloc[:, 2], ';')
    data.iloc[:, 2] = DataModel.remove_extra_characters(data.iloc[:, 2], '"')
    
    data = balanced_data(data, 2000)

    print('Dados depois do pré-processamento:')
    DataModel.print_data_info(data, data.iloc[:, 2])

    return data


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