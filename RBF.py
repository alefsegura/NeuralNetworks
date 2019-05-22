import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import random
from collections import namedtuple
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


""" readDataset: Function that reads and randomize the dataset, returning a namedtuple -> dataset.X and dataset.Y """
def readDataset(filename, y_columns):
    
    # Reading the dataset.
    data = pd.read_csv(filename, index_col=False, header=None, sep='\t')
    
    # Acquiring dataset data and class data.
    y = data.iloc[:,-1]
    y = np.array(y)
    X = data.iloc[:,:-1]
    X = np.array(X)
    
    # Randomizing dataset.
    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]
    
    # Creating an alias to dataset -> dataset.X and dataset.Y
    dataset = namedtuple('datset', 'X Y')

    return dataset(X=X_values, Y=y_values)


""" Função processing: Function that transform divides the dataset in train and test and transform the Y values in binary """
def processing(dataset, percentage):
    
    # Normalizing process.
    scaler = StandardScaler()
    scaler.fit(dataset.X)
    x = scaler.transform(dataset.X)
    
    # Labelizing process.
    onehot_encoder = OneHotEncoder(sparse=False)
    y = dataset.Y.reshape(len(dataset.Y), 1)
    y = onehot_encoder.fit_transform(y)

    # Computing the lenght of dataset.
    lenght = dataset.X.shape[0]

    # Split dataset into train and test.
    x_train = x[0:int(percentage*lenght), :]
    y_train = y[0:int(percentage*lenght), :]
    x_test = x[int(percentage*lenght):, :]
    y_test = y[int(percentage*lenght):, :]
        
    # Creating an alias to train and test set.
    dataset = namedtuple('datset', 'X Y')
    train = dataset(X=x_train, Y=y_train)
    test = dataset(X=x_test, Y=y_test)

    return train, test


""" Função hidden_train: função responsável por calcular a distância para cada cluster e aplicar a função gaussiana de base radial """
def hidden_train(x, n_clusters, clusters, sigma):
    
    distances = np.zeros(n_clusters)

    for i in range(n_clusters):
        
        # Calculando a distância euclidiana para cada cluster i.
        distances[i] = np.sqrt(np.sum(((x - clusters[i])**2)))
    
    # Applying radial basis function
    distances = np.exp(-(((distances**2))/(2*(sigma**2))))
    
    # Envia para a camada escondida o cálculo da distância do exemplo x para cada cluster.
    return distances


""" Função forward: função responsável por aplicar os pesos da camada de saida juntamente com as distancias obtidas para cada cluster """
def forward(x, output_weights):
    
    # Função de ativação = função identidade. Concatena com 1 p/ adição do theta.
    x_aux = np.concatenate((x, np.ones(1)))
    return np.matmul(x_aux, output_weights.T) 


""" Função backward: função responsável pela atualização dos pesos da camada de saída """
def backward(dataset, eta, n_classes, output_weights, entry, f_net_o, distance):
    
    y = dataset.Y[entry]
    
    # Calcula o erro
    error = y - f_net_o
    
    # Atualiza os pesos da camada de saída
    output_weights += eta*np.matmul(error.reshape(n_classes,1),np.append(distance,1).reshape(1,n_classes+1))

    return output_weights, error


""" Função testing: função responsável pela etapa de teste da rede RBF """
def testing(dataset, output_weights, n_clusters, clusters, sigma):
    counter = 0
    
    # Para cada entrada de teste..
    for entry in range(dataset.X.shape[0]):
        
        # Calcula a distancia para cada cluster
        distances = hidden_train(dataset.X[entry], n_clusters, clusters, sigma)
        
        # Aplica as distancias juntamente com os pesos obtidos
        y_hat = forward(distances, output_weights)
        
        # Computa a saida esperada e a obtida
        y_hat = np.argmax(y_hat)
        y = np.argmax(dataset.Y[entry])
        
        # Compara as saidas, se igual -> soma 1 acerto.
        if (y == y_hat):
            counter += 1

    return (counter/dataset.X.shape[0])


""" Função RBF: função responsável por todas as etapas de execução da rede RBF """
def RBF(dataset, n_classes, eta, data_ratio, epochs, sigma):
    train, test = processing(dataset, data_ratio)
    n_clusters = n_classes
    clusters = np.zeros((n_clusters, dataset.X.shape[1]))

    # Calculating the clusters
    for i in range(1, n_clusters+1):
        clusters[i-1] = train.X[np.argmax(train.Y,axis=1)+1 == i].mean(axis=0)
    
    hidden_units = n_clusters

    # Initializing and filling the weights values of output layer
    output_weights = np.zeros((n_clusters,n_classes+1))

    # Inicialização dos pesos da camada de saída com distribuição uniforme de -1 a 1
    for i in range(n_clusters):
        for j in range(n_classes+1):
            output_weights[i][j] = random.uniform(-1, 1)

    # Treina a rede de acordo com o número de épocas
    for i in range(epochs):
        
        error = 0

        # Para cada entrada de treino
        for entry in range(train.X.shape[0]):
            
            # Calcula a distancia do exemplo para cada cluster
            distances = hidden_train(train.X[entry], n_clusters, clusters, sigma)
            
            # Calcula a saida esperada com base nos pesos atuais
            f_net_o = forward(distances, output_weights)
            
            # Atualiza os pesos e computa o erro
            output_weights, erro = backward(train, eta, n_classes, output_weights, entry, f_net_o, distances)
            error += sum(erro*erro)

    # Etapa de teste
    return testing(test, output_weights, n_clusters, clusters, sigma)


# Lendo dataset
dataset = readDataset('seeds_dataset.txt', 1)

# Teste das acurácias da rede RBF
rbf_acc = []
for i in range(10):
    rbf_acc.append(RBF(dataset, 3, 0.01, 0.7, 500, 1.9))

# Resultado
print('Acurácias da Rede RBF:', rbf_acc)
