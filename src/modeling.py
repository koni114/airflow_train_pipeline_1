from datetime import datetime

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from config import config as conf
from config import set_env_minIO


def modeling(x: pd.DataFrame, y: pd.DataFrame):
    
    set_env_minIO(conf)

    tracking_uri = conf["mlflow_tracking_uri"]
    experiment_name = conf["proj_name"]

    mlflow.set_tracking_uri(tracking_uri)  
    mlflow.set_experiment(experiment_name)
    
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.4, random_state=2022)
    
    with mlflow.start_run():
        mlflow.autolog()

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train) 

        run_name = f"sklearn_knn_k_{str(3)}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" 
        
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.sklearn.log_model(sk_model=knn, artifact_path="models")
    
    mlflow.end_run()