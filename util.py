import numpy as np
from collections import namedtuple

def prepara_dataset(data, y_collumns, sep=','):
    
    y = data.iloc[:,len(data.columns)-y_collumns: len(data.columns)]
    y = np.array(y)
    X = data.iloc[:,0:len(data.columns)-y_collumns]
    X = np.array(X)
    
    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]
    
    dataset = namedtuple('datset', 'X Y')

    return dataset(X=X_values, Y=y_values)

def sample(dataset):
    sample = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
    for i in range(len(dataset.Y)):    
        sample[str(np.argmax(dataset.Y[i]))].append(dataset.X[i])
    return sample