import numpy as np
import matplotlib.pyplot as plt
from model import *

def plotDecisionBoundary(X_orig, Y, model, D=5, save_path="result.png"):
    x1_min, x1_max = X_orig[:, 0].min() - 0.1, X_orig[:, 0].max() + 0.1
    x2_min, x2_max = X_orig[:, 1].min() - 0.1, X_orig[:, 1].max() + 0.1

    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 300), np.linspace(x2_min, x2_max, 300))
    
    grid = np.c_[x1.ravel(), x2.ravel()]
    grid_mapped = featureMapping(grid, D)
    probs = model.predict(grid_mapped).reshape(x1.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(x1, x2, probs, alpha=0.4, cmap=plt.cm.RdYlGn)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=Y, cmap=plt.cm.RdYlGn, edgecolors='k')
    plt.title("Decision Boundary of coners")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
