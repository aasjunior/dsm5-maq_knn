from model.KNNModel import KNNModel
import numpy as np

def variable_k(model, knn_values):
    try:
        X = np.array(model.data.iloc[:, 1:len(model.numeric_cols)]) 
        Y = np.array(model.data[model.categorical_cols[0]]) 
        
        for k in knn_values['neighboors']:
            knn = KNNModel(model.data, k)
            accuracy = knn.train_and_evaluate(X, Y, knn_values['test_size'], knn_values['train_size'])

            print(f'Para k={k}, a acurácia foi de {accuracy}')
        
    except Exception as e:
        raise Exception(f'Erro na execução de variação de vizinhos: {e}')