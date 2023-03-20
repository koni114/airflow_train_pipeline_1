import pandas as pd
from sklearn import preprocessing


def preprocess(data: pd.DataFrame):
    
    y_col_name = ["variety"]

    label_encoder = preprocessing.LabelEncoder()
    data[y_col_name] = pd.DataFrame(label_encoder.fit_transform(data[y_col_name]), columns=y_col_name)

    y = data[y_col_name]
    x = data.drop(y_col_name, axis=1)
   
    return x, y