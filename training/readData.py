import pandas as pd
import numpy as np

def read(filename):
    df = pd.read_csv(filename, sep=' ', delimiter=' ', header=None)
    df = df.values
    features,labels = np.split(df,[256], axis=1)
    labels = np.delete(labels,-1,1)
    return features,labels

