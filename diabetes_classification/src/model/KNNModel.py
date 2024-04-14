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