import pandas as pd

class DataModel:
    def __init__(self, file_path, numeric_cols, categorical_cols):
        self.file_path = file_path
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.data = self.load_data()
        self.data.dropna(inplace=True)

    def load_data(self):
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            return f'O arquivo {self.file_path} não foi encontrado.'
        except pd.errors.ParserError:
            return f'Ocorreu um erro ao analisar o arquivo {self.file_path}.'
        except Exception as e:
            return f'Ocorreu um erro: {e}'
    