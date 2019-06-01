import sys
import os
import math
import numpy as np
from sklearn.cluster import KMeans
PATH = os.path.split(os.path.abspath(__file__))[0]
PARENT_PATH = os.path.dirname(PATH)
sys.path.append(PARENT_PATH)
from training.readData import read


def generateClusters(features, labels, num_clusters):
    #vetor que separa os dados por classe
    separated_features = [list() for k in range(10)]
    for i in range(len(labels)):
        ind = np.where(labels[i]==1)[0][0]
        separated_features[ind].append(features[i])
    matrix_centers = list()
    for sep_class in separated_features:
        #adiciona todos os vetores de centro a uma matriz
        matrix_centers.extend(KMeans(num_clusters).fit(sep_class).cluster_centers_)
    return matrix_centers

def sigmoid(result):
    return (1/(1+math.exp(-result)))
vsigmoid = np.vectorize(sigmoid)


def output_centers(matrix_centers, vector_values, beta):
    output = list()
    for line in matrix_centers:
        #aux = np.exp(-(np.linalg.norm(np.subtract(vector_values,line)))*(np.linalg.norm(np.subtract(vector_values,line)))/beta)
        output.append(np.exp(-(np.linalg.norm(np.subtract(vector_values, line))**2)/beta))
    return(np.array(output))

def randomize(features, labels):
    data = np.array(list(zip(features,labels)))
    np.random.shuffle(data)
    features, labels = zip(*data)
    return np.array(features), np.array(labels)

def forward(feature, weights, matrix_centers, beta):
    h = output_centers(matrix_centers, feature, beta)
    #add bias
    h =  np.append(h, np.array([1]))
    y = np.matmul(weights, h.reshape((-1,1)))
    y = vsigmoid(y)
    return y,h

def backpropagation(N_ITER, features, labels, matrix_centers, num_clusters, ETA, beta):
    #create random init weight matrix
    weights = np.random.randn(10, 10*num_clusters + 1)*0.01
    for iteration in range(N_ITER):
        for i in range(len(features)):
            y, h = forward(features[i], weights, matrix_centers, beta) 
            h = np.transpose(h.reshape((-1,1)))
            erro = (y - labels[i].reshape((-1,1)))
            gradient_value = erro*y*(1-y)
            weights = weights - ETA*np.matmul(gradient_value, h)
    return weights

def train(num_clusters=50, eta=0.1, n_iter=10, beta=150, cross_validation=False, features=None, labels=None):
    if(not cross_validation):
        features, labels = read(os.path.join(PARENT_PATH,"data","semeion.txt"))
    matrix_centers = generateClusters(features, labels, num_clusters)
    features, labels = randomize(features, labels)
    weights = backpropagation(n_iter, features, labels, matrix_centers, num_clusters, eta, beta)
    
    if(not cross_validation):
        np.save(os.path.join(PARENT_PATH,"data","weights"), weights)
        np.save(os.path.join(PARENT_PATH,"data","centers"), matrix_centers)
    return weights, matrix_centers

