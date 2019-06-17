import os
import sys
import numpy as np
import pandas as pd

PATH = os.path.split(os.path.abspath(__file__))[0]
PARENT_PATH = os.path.dirname(PATH)
sys.path.append(PARENT_PATH)
from training.training import train, forward


class RBFN():
    
    def __init__(self, beta=40):
        self.beta = beta
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def train(self, eta=0.4, num_clusters=40, n_iter=15):
        self.eta = eta
        self.num_clusters = num_clusters
        self.n_iter = n_iter
        try:
            self.weights, self.matrix_centers = train(num_clusters=self.num_clusters, eta=self.eta, 
                    n_iter=self.n_iter,beta=self.beta, cross_validation=True, features=self.X,labels=self.y)
        except:
            self.weights, self.matrix_centers = train(num_clusters=self.num_clusters, eta=self.eta, n_iter=self.n_iter,beta=self.beta)

    def loadData(self):
        self.weights = np.load(os.path.join(PARENT_PATH,"data","weights.npy"))
        self.matrix_centers = np.load(os.path.join(PARENT_PATH,"data","centers.npy"))
    
    def predict(self, test):
        y, h = forward(test, self.weights, self.matrix_centers, self.beta)
        ind = np.where(y==np.amax(y))[0][0]
        return ind

    def readPredict(self, input_file):
        df = pd.read_csv(input_file, sep=' ', delimiter=' ', header=None)
        df = df.values
        for line in df:
            print(self.predict(line[:256]))

