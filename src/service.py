#!/usr/bin/env python
import os
import sys

SRC_HOME = os.path.dirname(os.path.realpath(__file__))
if SRC_HOME not in sys.path:
    sys.path.append(SRC_HOME)
    
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, Text

import config as conf


## laod_model, load_service 함수는 수정하시지 않으셔도 됩니다.
def load_model(model_name:str)->bentoml.Runner:
    model_runner = bentoml.models.get(f"{model_name}").to_runner()
    return model_runner

def load_service(model_name:str, runners:bentoml.Runner)->bentoml.Service:
    svc = bentoml.Service(model_name, runners=[model_runner])
    return svc

model_name = conf['model_name']

model_runner = load_model(model_name)
svc = load_service(model_name, model_runner)



CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
FEATURES_NAME = ['sepal length', 'sepal width', 'petal length', 'petal width']

@svc.api(
    """
    모델에 들어오는 Input/Output에 대한 타입을 정의 할 수 있습니다.
    해당 예시에서는 numpy 형태의 Input에 대해, Text 형태의 Output을 출력하는 사례 입니다.
    
    "NumpyNdarray.from_sample" method 는 사용자가 아무것도 입력하지 않았을 경우,
    해당 [4.9, 3.0, 1.4, 0.2] 예시로 보이도록 하여, 입력 차원을 고려할 수 있도록 한 것입니다.
    """
    input=NumpyNdarray.from_sample(np.array([4.9, 3.0, 1.4, 0.2], dtype=np.double)),
    output=Text(),
)

def _predict(input_feature: np.ndarray) -> str:
    """
    해당 함수는 위에서 정의한 Input/Output에 대해 매핑시키고 실제 모델이 추론하는 코드 입니다.
    
    input인 "input_feature"를 np.ndarray 로 받도록 하고, 해당 input이 모델에 들어갑니다.
    모델에서 추론된 결과가 CLASS_NAMES 와 매핑하여 output 타입인 str 형태로 출력되도록 합니다.
    
    원하시는 데이터 타입은 Service.api 에서 먼저 정의하고 매핑하시어 활용하시면 됩니다.
    """
    results = model_runner.predict.run([input_feature])
    result = results[0]
    category = CLASS_NAMES[result]
    return category