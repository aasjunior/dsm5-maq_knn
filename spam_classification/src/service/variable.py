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