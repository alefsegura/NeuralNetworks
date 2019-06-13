import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import namedtuple

class RBFNet:

    def __init__(self, eta, sigma, hidden_units):
        # Instanciando principais parâmetros para o modelo
        self.eta = eta
        self.sigma = sigma
        self.hidden_units = hidden_units
        
    def train_test_split(self, dataset, train_size):
        
        # Computing the lenght of dataset
        lenght = dataset.X.shape[0]

        # Split dataset into train and test.
        x_train = dataset.X[0:int(train_size*lenght),:]
        y_train = dataset.Y[0:int(train_size*lenght),:]
        x_test = dataset.X[int(train_size*lenght):,:]
        y_test = dataset.Y[int(train_size*lenght):,:]
        
        # Creating an alias to train and test set.
        dataset = namedtuple('datset', 'X Y')
        train = dataset(X=x_train, Y=y_train)
        test = dataset(X=x_test, Y=y_test)
        
        return train, test
 
    def distances(self, x, n_clusters, clusters, sigma):
        distances = np.zeros(n_clusters)
        for i in range(n_clusters):

            # Calculando a distância euclidiana para cada cluster i
            distances[i] = np.sqrt(np.sum(((x - clusters[i])**2)))
        
        # Aplicando a radial basis function
        return np.exp(-(((distances**2))/(2*(sigma**2)))) 

    def forward(self, x, output_weights):
        x_aux = np.concatenate((x, np.ones(1)))       
        return np.matmul(x_aux, output_weights) 

    def backward(self, y, eta, n_classes, output_weights, f_net_o, distance, n_clusters):
        # Calcula o erro
        error = y - f_net_o        

        # Atualiza os pesos da camada de saída
        output_weights += eta*(np.matmul(error.reshape(n_classes,1), np.append(distance,1).reshape(1,n_clusters+1)).T)
        
        return output_weights, sum(error*error)
 
    def fit(self, dataset, n_classes, clusters, train_size, delta_error, verbose=False):
        eta = self.eta 
        sigma = self.sigma
        n_clusters = self.hidden_units
            
        train, test = self.train_test_split(dataset, train_size)

        # Inicialização dos pesos da camada de saída com distribuição uniforme de -1 a 1
        output_weights = np.zeros((n_clusters+1,n_classes))
        for i in range(n_clusters+1):
            for j in range(n_classes):
                output_weights[i][j] = random.uniform(-1, 1)
        
        # Início das épocas de aprendizagem
        delta = delta_error + 1
        errors_list = [1,0]
        if verbose:
            print('Epoch | Erro')
            
        # Enquanto o delta não dinuir menos que o hiperparâmetro delta_error, haverá uma nova época
        epoch = 0
        while(abs(delta) > delta_error):
            sum_errors = 0
            
            # Para cada entrada de treino
            for entry in range(train.X.shape[0]):

                # Calcula a distancia da entrada para cada cluster
                distances = self.distances(train.X[entry], n_clusters, clusters, sigma)
                
                # Calcula a saida esperada com base nos pesos atuais
                f_net_o = self.forward(distances, output_weights)
                
                # Atualiza os pesos e computa o erro
                output_weights, error = self.backward(train.Y[entry], eta, n_classes, output_weights, f_net_o,
                                                      distances, n_clusters)
                sum_errors += error
            
            if verbose:
                print(epoch, sum_errors)
                
            errors_list.append(sum_errors)
            delta = errors_list[-1] - errors_list[-2]
            epoch += 1
            
        if not verbose:
            print('Last epoch: {} | Error: {}'.format(epoch, error))

        # Salva os parâmetros aprendidos no objeto
        self.output_weights = output_weights
        self.clusters = clusters
        self.n_clusters = n_clusters
        self.dataset = dataset

    def score(self):
        # Recupera os parâmetros aprendidos
        dataset = self.dataset

        # Para cada entrada de teste
        counter = 0
        for entry in range(dataset.X.shape[0]):
            y_hat = self.predict(dataset.X[entry])
            y = np.argmax(dataset.Y[entry])

            # Compara as saidas, se igual -> soma 1 acerto
            if (y == y_hat):
                counter += 1

        return (counter/dataset.X.shape[0])

    def predict(self, entry):
        # Recupera os parâmetros aprendidos
        output_weights = self.output_weights
        n_clusters = self.n_clusters
        clusters = self.clusters
        sigma = self.sigma

        # Calcula o valor previsto para a entrada
        distances = self.distances(entry, n_clusters, clusters, sigma)
        y_hat = self.forward(distances, output_weights)

        return np.argmax(y_hat)

