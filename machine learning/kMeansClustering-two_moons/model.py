import numpy as np

class KernelKMeans:
    def __init__(self, cluster_k, iteration):
        self.cluster_k = cluster_k # 클러스터 개수
        self.iteration = iteration # cluster 업데이트 시 최대 반복횟수
        self.assign = None
    
    def RBFKernel(self, X): # 커널 계산
        m = X.shape[0]
        kn = np.zeros((m, m))
        gamma = 2 * 0.5
        for i in range(m):
            for j in range(m):
                distance = X[i] - X[j]
                square_dis = np.dot(distance, distance)
                kn[i, j] = np.exp(-1 * 1/gamma * square_dis)
        return kn
    
    def learning(self, X):
        m = X.shape[0]
        kn = self.RBFKernel(X) # 커널 계산
        self.assign = np.random.randint(self.cluster_k, size=m)
        # 샘플에 랜덤으로 클러스터 할당(초기화)

        for _ in range(self.iteration):
            prev = self.assign.copy() # 이전 클러스터 할당
            self.updateClusters(kn)

            if np.all(prev == self.assign): # 클러스터 수렴 시 조기종료
                break
    
    def updateClusters(self, kn): 
        m = kn.shape[0]
        distance = np.zeros((m, self.cluster_k))
        # distance: feature space에서 샘플과 클러스터 중심 값 사이의 거리 저장
        
        for k in range(self.cluster_k):
            members = [i for i in range(m) if self.assign[i] == k]
            # member: cluster k에 속한 샘플
            count = len(members)

            if count == 0: # cluster k에 아무것도 없는 경우
                distance[:, k] = np.inf
                continue

            sum_uk = 0
            for a in members:
                for b in members:
                    sum_uk += kn[a, b]
            t3 = sum_uk / (count ** 2) # K(u_k, u_k)

            for i in range(m):
                t1 = kn[i, i] # K(x_n, x_n)

                sum_xn = 0
                for j in members:
                    sum_xn += kn[i, j]
                t2 = -2 * (sum_xn / count) # -2K(x_n, u_k)

                distance[i, k] = t1 + t2 + t3 
                # feature space에서 x_n과 u_k의 유클리드 거리 제곱
        self.assign = np.argmin(distance, axis = 1)
        # 샘플 할당 업데이트
