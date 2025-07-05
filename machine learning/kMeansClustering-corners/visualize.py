import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(X, cluster, cluster_k, save_path="result.png"):
    plt.figure(figsize=(6, 6))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    for i in range(cluster_k):
        cluster_points = X[cluster == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
        color = colors[i % len(colors)], label=f"Cluster {i}", edgecolors='k')

    plt.title("Kernel K-Means Clustering Result of corners")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()