from model.DataModel import DataModel
from service.variable import variable_k

try:
    knn_values = {
        'neighboors': [3, 5, 7],
        'test_size': 0.3,
        'train_size': 0.7
    }

    numeric_cols = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']
    categorical_cols = ['Diabetes_binary']

    model = DataModel('db/diabetes_binary.data', numeric_cols, categorical_cols)

    variable_k(model, knn_values)

except Exception as e:
    print(f'Ocorreu um erro:\n{e}')