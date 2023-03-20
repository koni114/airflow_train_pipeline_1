import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    
    iris = load_iris()

    x_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    y_columns = ["variety"]

    x = pd.DataFrame(iris.data, columns=x_columns)
    y = pd.DataFrame(iris.target, columns=y_columns)

    data = pd.concat([x, y], 1)
    
    return data