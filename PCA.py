import numpy as np      # effective math
import matplotlib.pyplot as plt     # ultimate plotting tool
from mpl_toolkits.mplot3d import Axes3D     # 3D plots
import pandas as pd     # allow us to make dataframes to store our data cleanly
from sklearn import datasets  

class PCA:
  def __init__(self):
    pass
  
  def fit(self, X):
    """
    This is the fit method
    """
    X -= np.mean(X, axis=0)
    X /= np.ptp(X, axis=0)
    cov = X.T @ X
    eigenvalues, eigenvector = np.linalg.eig(cov)

if __name__ == "__main__":
  data = pd.read_csv('../DATA/iris.csv')

  label_dict = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}  # dictionary containing label to number mapping
  data["variety"] = data["variety"].replace(label_dict)

  X = data[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
  Y = data["variety"]

  print(X.shape)      # 150 rows (datapoints), 4 columns (features)
  print(Y.shape)      # 150 single dimension labels

  m = X.shape[0]      # 150 rows

  pca = PCA()
  pca.fit(X)



