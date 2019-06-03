import os
import sys
import numpy as np

PATH = os.path.split(os.path.abspath(__file__))[0]
PARENT_PATH = os.path.dirname(PATH)
sys.path.append(PARENT_PATH)

from cross_validation import CV

def search():
    acuracia = 0
    for eta in np.arange(0.05, 4.0, 0.05):
        for beta in range(50, 310, 50):
            for n_iter in range(10, 50, 10):
                for num_clusters in range(10, 100, 10):
                    result=CV(k=5, eta=eta, beta=beta, num_clusters=num_clusters, n_iter=n_iter)
                    if(result>acuracia):
                        acuracia = result
                        best_eta = eta
                        best_beta = beta
                        best_num_clusters = num_clusters
                        best_n_iter = n_iter
                        print("\nacuracia: ", str(acuracia))
                        print("\neta: ", str(best_eta))
                        print("\nbeta: ", str(best_beta))
                        print("\nn_clusters: ", str(best_num_clusters))
                        print("\nn_iter: ", str(best_n_iter))
                        print("----------------")
#search()
