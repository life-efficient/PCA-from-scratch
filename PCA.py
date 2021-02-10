import numpy as np      # effective math
import matplotlib.pyplot as plt     # ultimate plotting tool
from mpl_toolkits.mplot3d import Axes3D     # 3D plots
import pandas as pd     # allow us to make dataframes to store our data cleanly
from sklearn import datasets  

class PCA:
  def __init__(self, n_com):
      self.n_com = n_com
  
  
  def fit(self, X):
    """
    This is the fit method
    """
    m = len(X)
    X -= np.mean(X, axis=0)
    X /= m
    cov = X.T @ X
    eigenvalues, eigenvector = np.linalg.eig(cov)
    print(eigenvector)
    print(eigenvalues)
    X = X @ eigenvector
    return X[:,:self.n_com]

if __name__ == "__main__":
  data = datasets.load_iris()
  X = data['data']
  Y = data['target']

  print(X.shape)      # 150 rows (datapoints), 4 columns (features)
  print(Y.shape)      # 150 single dimension labels

  m = X.shape[0]      # 150 rows

  pca = PCA(n_com=2)
  x_transfrom = pca.fit(X)
  print(x_transfrom)



