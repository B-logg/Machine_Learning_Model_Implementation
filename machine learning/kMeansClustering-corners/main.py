import numpy as np
from model import *
from visualize import *
import os

def getData(dataset_name):
    current_dir = os.path.dirname(__file__) 
    # 현재 디렉토리를 기준으로 경로 설정
    base_path = os.path.join(current_dir, 'datasets', dataset_name)
    X = np.loadtxt(os.path.join(base_path, 'dataX.txt'))
    return X

if __name__ == "__main__":
    dataset_name = 'corners'

    X = getData(dataset_name)

    model = KernelKMeans(cluster_k=2, iteration=100)
    model.learning(X)

    prediction = model.assign
    print("\n[예측]\n", prediction)

    plot_clusters(X, model.assign, model.cluster_k, save_path = f'KMeans_result_{dataset_name}.png')
