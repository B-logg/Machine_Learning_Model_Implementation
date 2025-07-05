import numpy as np
import matplotlib.pyplot as plt
from model import *
from visualize import *
import os

def getData(dataset_name):
    current_dir = os.path.dirname(__file__) 
    # 현재 디렉토리를 기준으로 경로 설정
    base_path = os.path.join(current_dir, 'datasets', dataset_name)
    X = np.loadtxt(os.path.join(base_path, 'dataX.txt'))       # (m, 2)
    Y = np.loadtxt(os.path.join(base_path, 'dataY.txt')).astype(int)  # (m,)
    return X, Y

if __name__ == "__main__":
    dataset_name = 'corners'
    X, Y = getData(dataset_name) # 데이터 셋 불러오기

    D = 5 # feature mapping 차원 설정
    X_mapped = featureMapping(X, D)

    bestLr, bestEpoch = crossValidation(X_mapped, Y, k = 5)
    print("Best Learning Rate: {bestLr}")
    print("Best Epoch: {bestEpoch}")
    # 검증 데이터셋을 통해 최적의 learning rate와 epoch 구하기

    model = LogisticRegression(a=bestLr, epoch = bestEpoch)
    model.learning(X_mapped, Y)
    # 모델 학습 시키기

    prediction = model.predict(X_mapped)
    print("\n[예측]\n", prediction)

    print("\n[Model Parameters]\n", model.weight)
    # 학습된 모델 파라미터 출력하기

    print(f"Best Learning Rate: {bestLr}")
    print(f"Best Epoch: {bestEpoch}")

    plotDecisionBoundary(X, Y, model, D, save_path=f'LR_result_{dataset_name}.png')
    # 학습된 decision boundary 시각화하기.