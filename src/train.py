#!/usr/bin/env python
'''
아래 import 한 함수들은 예시입니다.
학습 파이프라인 구성을 위하여 자유롭게 import 하여 사용할 수 있습니다.
'''
from load_data import load_data
from modeling import modeling
from preprocess import preprocess


def train():
    """
        학습 파이프라인에 필요한 함수들을 호출하여 train 함수를 구성해주세요.
        import 할 script 는 src 디렉토리 하위에 위치하여야 합니다.
        임의로 train 함수 명은 변경하여서는 안되니 주의해주세요.
    """
    data = load_data()

    # 2. data preprocessing
    x, y = preprocess(data=data)

    # 3. modeling
    modeling(x=x, y=y)

if __name__ == "__main__":
    """
    학습 파이프라인 구성시, 반드시 if __name__ == "__main__" 구문을 통해서 train 함수를 호출하여야 합니다.
    이 때, train 함수 명은 변경하면 안되니 주의해주세요.
    """
    train()