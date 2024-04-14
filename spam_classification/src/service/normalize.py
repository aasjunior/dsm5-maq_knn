from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load, dump
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
        print(f'Ocorreu um erro na normalização dos dados: {e}')


def encode_labels(model):
    try:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(model['Category'])
    
        return labels

    except Exception as e:
        print(f'Ocorreu um erro na conversão dos rótulos: {e}')


def preprocess_message(message):
    try:
        # Remove non-alphabetic characters
        message = re.sub('[^a-zA-Z]', ' ', message)
        
        # Convert to lower case
        message = message.lower()
        
        return message
    
    except Exception as e:
        print(f'Ocorreu um erro no pré-processamento dos dados: {e}')


def save_data(data_normalized):
    try:
        with open('db/normalized_data.txt', 'wb') as filehandle:
            dump(data_normalized, filehandle)

    except Exception as e:
        print(f'Ocorreu um erro ao tentar salvar o arquivo: {e}')
