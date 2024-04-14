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