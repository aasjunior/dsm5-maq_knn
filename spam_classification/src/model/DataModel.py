import pandas as pd

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
        
        except pd.errors.ParserError:
            raise Exception(f'Ocorreu um erro ao analisar o arquivo {self.file_path}.')
        
        except Exception as e:
            raise Exception(f'Ocorreu um erro ao tentar ler o arquivo: {e}')
    

    def normalize_data(self):
        try:
            data = self.load_data()
            data.dropna(inplace=True)
            data = data.drop_duplicates()
            # duplicated = data.duplicated().sum()

            return data
        
        except Exception as e:
            raise Exception(f'Ocorreu um erro na normalização: {e}')