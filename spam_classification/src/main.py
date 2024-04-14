from model.DataModel import DataModel
from service.normalize import normalize_data, encode_labels
from service.variable import variable_k

knn_values = {
    'neighboors': [3, 5, 7],
    'test_size': 0.3,
    'train_size': 0.7
}

numeric_cols = ['Message']
categorical_cols = ['Category']

model = DataModel('db/spam_text_message.data', numeric_cols, categorical_cols)

variable_k(model.data, knn_values)