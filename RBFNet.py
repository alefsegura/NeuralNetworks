import warnings
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import namedtuple

class RBFNet:

    def __init__(self, eta, sigma):
        self.eta = eta
        self.sigma = sigma
        
    def train_test_split(self, dataset, train_size):
        # Computing the lenght of dataset.
        lenght = dataset.X.shape[0]
        # Split dataset into train and test.
        x_train = dataset.X[0:int(train_size*lenght), :]
        y_train = dataset.Y[0:int(train_size*lenght), :]
        x_test = dataset.X[int(train_size*lenght):, :]
        y_test = dataset.Y[int(train_size*lenght):, :]
        # Creating an alias to train and test set.
        dataset = namedtuple('datset', 'X Y')
        train = dataset(X=x_train, Y=y_train)
        test = dataset(X=x_test, Y=y_test)
        return train, test

    def hidden_train(self, x, n_clusters, clusters, sigma):
        distances = np.zeros(n_clusters)
        for i in range(n_clusters):
            # Calculando a distância euclidiana para cada cluster i.
            distances[i] = np.sqrt(np.sum(((x - clusters[i])**2)))
        # Applying radial basis function
        distances = np.exp(-(((distances**2))/(2*(sigma**2))))
        # Envia para a camada escondida o cálculo da distância do exemplo x para cada cluster.
        return distances

    def forward(self, x, output_weights):
        # Função de ativação = função identidade. Concatena com 1 p/ adição do theta.
        x_aux = np.concatenate((x, np.ones(1)))
        return np.matmul(x_aux, output_weights.T) 

    def backward(self, dataset, eta, n_classes, output_weights, entry, f_net_o, distance):
        y = dataset.Y[entry]
        # Calcula o erro
        error = y - f_net_o
        # Atualiza os pesos da camada de saída
        output_weights += eta*np.matmul(error.reshape(n_classes,1),np.append(distance,1).reshape(1,n_classes+1))
        return output_weights, error    
    
    def fit(self, dataset, n_classes, train_size, delta_error, verbose=False):   
        eta = self.eta 
        sigma = self.sigma

        train, test = self.train_test_split(dataset, train_size)
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
        
        # Epochs
        if verbose:
            print('Epoch | Erro')

        epoch = 0
        old_delta = 10
        delta = 0
        
        while(abs(delta-old_delta) > delta_error):
            error = 0
            # Para cada entrada de treino
            for entry in range(train.X.shape[0]):
                # Calcula a distancia do exemplo para cada cluster
                distances = self.hidden_train(train.X[entry], n_clusters, clusters, sigma)
                # Calcula a saida esperada com base nos pesos atuais
                f_net_o = self.forward(distances, output_weights)
                # Atualiza os pesos e computa o erro
                output_weights, erro = self.backward(train, eta, n_classes, output_weights, entry, f_net_o, distances)
                error += sum(erro*erro)
            epoch += 1
            if verbose:
                print(epoch,error)
            old_delta = delta
            delta = error
        
        if not verbose:
            print('Last epoch: {} | Error: {}'.format(epoch, error))

        self.output_weights = output_weights
        self.clusters = clusters
        self.n_clusters = n_clusters
        self.dataset = dataset

    def predict(self, entry):
        output_weights = self.output_weights
        n_clusters = self.n_clusters
        clusters = self.clusters
        sigma = self.sigma
        distances = self.hidden_train(entry, n_clusters, clusters, sigma)
        y_hat = self.forward(distances, output_weights)
        return np.argmax(y_hat)
        
    def score(self):
        dataset = self.dataset
        output_weights = self.output_weights
        n_clusters = self.n_clusters
        clusters = self.clusters
        sigma = self.sigma
        counter = 0
        # Para cada entrada de teste..
        for entry in range(dataset.X.shape[0]):
            y_hat = self.predict(dataset.X[entry])
            y = np.argmax(dataset.Y[entry])
            # Compara as saidas, se igual -> soma 1 acerto.
            if (y == y_hat):
                counter += 1
        return (counter/dataset.X.shape[0])

