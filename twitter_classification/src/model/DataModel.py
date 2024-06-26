import pandas as pd
import csv
from traceback import format_exception

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
        
        except pd.errors.ParserError as e:
            try:
                return self.load_by_line()
            
            except Exception as e:
                raise Exception(f'Erro ao analisar o arquivo {self.file_path}.\n\nTraceback: {e}')
        
        except Exception as e:
            raise Exception(f'Erro ao tentar ler o arquivo: {e}')
    

    def normalize_data(self):
        try:
            data = self.load_data()
            self.print_data_info(data, data.iloc[:, 3])
            
            data.dropna(inplace=True)
            data = data.drop_duplicates()

            # duplicated = data.duplicated().sum()

            return data
        
        except Exception as e:
            raise Exception(f'Erro na normalização: {e}')
        

    def load_by_line(self):
        error_count = 0
        data_list = []

        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                try:
                    if len(row) != len(headers):
                        raise ValueError('Número de campos na linha não corresponde ao número de campos no cabeçalho')
                    
                    data_list.append(row)
                    
                except Exception as e:
                    error_count += 1
                    continue
        
        print(f'Carregamento do arquivo {self.file_path}:\nNúmero de linhas com erro: {error_count}\n\n')
    
        return pd.DataFrame(data_list, columns=headers)


    @staticmethod
    def print_data_info(data, label_column):
        print(f'Número de instâncias: {data.shape[0]}')
        print(f'Número de atributos: {data.shape[1]}')
        print(f'Nome dos atributos: {data.columns.tolist()}')
        print(f'Número de linhas duplicadas: {data.duplicated().sum()}')
        print(f'Número de linhas com dados faltantes: {data.isnull().sum().sum()}')

        if label_column.dtype == 'object':
            print(f'\nClasses na coluna {label_column.name}:')
            print(label_column.value_counts())
            print('\n')


    @staticmethod
    def remove_extra_characters(data_column, character):
        return data_column.str.replace(character, '')