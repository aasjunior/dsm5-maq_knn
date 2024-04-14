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

    model = DataModel('db/tweets.data', numeric_cols, categorical_cols)

    variable_k(model.data, knn_values)

except Exception as e:
    print(f'Ocorreu um erro:\n{e}')