import os
import sys
import numpy as np

PATH = os.path.split(os.path.abspath(__file__))[0]
PARENT_PATH = os.path.dirname(PATH)
sys.path.append(PARENT_PATH)

from models.models import RBFN
from training.readData import read
from training.training import randomize

def CV(k=10, eta=0.1, beta=150, num_clusters=50, n_iter=10):
    features, labels = read(os.path.join(PARENT_PATH, "data", "semeion.txt"))
    data = np.array(list(zip(features,labels)))
    np.random.shuffle(data)
    data = np.array_split(data, k)
    acerto = 0
    erro = 0
    for i in range(k):
        data_training = np.concatenate([data[j][:] for j in range(k) if j!=i])
        features, labels = zip(*data_training)
        X_test, y_train = zip(*data[i])
        model = RBFN(beta)
        model.fit(features,labels)
        model.train(eta=eta, num_clusters=num_clusters, n_iter=n_iter)
        erro_fold = 0
        acerto_fold = 0
        for w in range(len(X_test)):
            y_test = model.predict(X_test[w])
            y_ind = np.where(y_train[w]==1)[0][0]
            if(y_test==y_ind):
                acerto_fold = acerto_fold + 1
            else:
                erro_fold = erro_fold + 1
        acerto = acerto + acerto_fold
        erro = erro + erro_fold
        print("Acurácia do fold "+str(i)+": "+str(acerto_fold/(acerto_fold+erro_fold)))

    print("--------")
    print("Acurácia Total: "+str(acerto/(acerto+erro)))
CV(eta=0.05, n_iter=30)

