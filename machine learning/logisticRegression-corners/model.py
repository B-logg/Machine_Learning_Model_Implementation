import numpy as np
import math

def sigmoid(z): # 시그모이드 함수
    return 1 / (1 + np.exp(-z))

def featureMapping(X, D): # 특성변환
    # 현재 데이터 셋 x1, x2, 이차원 데이터
    m = X.shape[0]
    result = [np.ones(m)] # bias 추가

    for degree in range(1, D + 1):
        for i in range(degree + 1):
            result.append((X[:, 0] ** i) * (X[:, 1] ** (degree - i)))
            # (x1의 i제곱 x x2의 j제곱)
    return np.stack(result, axis=1) # 특성 변환된 여러 배열들을 열 방향으로 이어 붙이기


class LogisticRegression:
    def __init__(self, a, epoch):
        self.weight = None # parameter theta
        self.a = a # learning rate
        self.epoch = epoch # 반복횟수

    
    def gradientDescent(self, r, X, Y, a): # 경사하강법
        gradient = a * (np.dot(X.T, (r - Y))) # NLL의 Gradient * learning rate
        self.weight -= gradient # 파라미터 업데이트
        return self.weight

    def learning(self, X, Y): # 학습
        m, n = X.shape
        self.weight = np.zeros(n) # 파라미터 초기화
        for i in range(self.epoch):
            thetaX = np.dot(X, self.weight)
            r = sigmoid(thetaX)
            self.gradientDescent(r, X, Y, self.a)

            print(f"Epoch {i} Loss: {self.costFunction(X, Y)}")
    
    def costFunction(self, X, Y):
        m = X.shape[0]
        p = sigmoid(np.dot(X, self.weight))
        epsilon = 1e-15 # log(0)을 방지하기 위함
        cost = -1 * np.sum(Y * np.log(p + epsilon) + (1 - Y) * np.log(1 - p + epsilon))
        return cost
    
    def predictProb(self, X): # 확률 예측
        prob = sigmoid(np.dot(X, self.weight))
        return prob
        
    def predict(self, X): # 레이블 예측
        prob = self.predictProb(X)
        m = prob.shape[0]
        for i in range(m):
            if prob[i] >= 0.5 : prob[i] = 1
            else : prob[i] = 0
        return prob

        
def KFoldSplit(X, Y, k=5, seed = 10):
        np.random.seed(seed)
        m = X.shape[0]
        splitIndices = np.random.permutation(m)
        foldSize = m // k

        folds = []
        for i in range(k):
            validationSet = splitIndices[i * foldSize : (i + 1) * foldSize]
            trainSet = np.concatenate((splitIndices[:i * foldSize], splitIndices[(i + 1)*foldSize : ]))
            folds.append((trainSet, validationSet))
        
        return folds
    
def crossValidation(X, Y, k = 5):
    folds = KFoldSplit(X, Y, k=5, seed = 10)

    learningRate = [0.1, 0.01, 0.001, 0.0001]
    epoch = [10, 100, 1000, 10000]

    for i, (trainIdx, valIdx) in enumerate(folds):
        X_train, Y_train = X[trainIdx], Y[trainIdx]
        X_val, Y_val = X[valIdx], Y[valIdx]
        mini = math.inf
        bestLr = 1
        bestEpoch = 1
        for a in learningRate:
            for e in epoch:
                model = LogisticRegression(a, e)
                model.learning(X_train, Y_train)
                loss = model.costFunction(X_val, Y_val)
                if (mini > loss) : 
                    mini = loss
                    bestLr = a
                    bestEpoch = e
    return bestLr, bestEpoch
