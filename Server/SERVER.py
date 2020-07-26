#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:24:26 2018

@author: joel
"""

import os, subprocess, time
import json
import random

import pandas as pd
import numpy as np
#from pyDOE import lhs
from flask import Flask, request, jsonify

from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from keras.layers.core import Activation, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM
import keras

from numba import cuda
import gc

app = Flask(__name__)

#variaveis globais


"""
######################## Class for NeuralNetworks ########################
"""
class NeuralNetwork:
    def __init__(self, dim_1, dim_2, output_labels, classifier_name):
        self.session = tf.Session()

        self.graph = tf.get_default_graph()
        # the folder in which the model and weights are stored
        self.model_folder = os.path.join(os.path.abspath("src"), "static")
        self.model = None
        # for some reason in a flask app the graph/session needs to be used in the init else it hangs on other threads
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    self.model = Sequential()
                    hidden_layer_nodes = int(classifier_name.split('_')[1])
                    if classifier_name.startswith('RNN'):
                        self.model.add(SimpleRNN(hidden_layer_nodes, return_sequences=False, input_shape=(dim_1, dim_2)))
                    elif classifier_name.startswith('LSTM'):
                        self.model.add(LSTM(hidden_layer_nodes, return_sequences=False, input_shape=(dim_1, dim_2)))
                    # self.model.add(Dropout(0.2))
                    self.model.add(Dense(units=output_labels))
                    self.model.add(Activation('softmax'))
                    self.model.compile(loss='mean_squared_error', optimizer='adam')
                    # return True
                except Exception as e:
                    print(e)
                    # return False

        # Initialize all TF variables.
        self.session.run(tf.global_variables_initializer())

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y

    def fit(self, x, y, epochs, verbose, shuffle):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.fit(x, y, epochs=epochs, verbose=verbose, shuffle=shuffle)

"""
######################## INICIALIZA OS CLASSIFICADORES ##################################################################
"""
#classifierSVM = SVR(kernel='rbf', C=1e3, gamma=0.1, tol=0.01)
#classifierMLP = MLPRegressor(hidden_layer_sizes=(10,),max_iter=1000)
# #                                      activation='relu',
# #                                      solver='adam',
#  #                                     learning_rate='adaptive',
#   #                                    max_iter=10000,
#    #                                   learning_rate_init=0.01,
#     #                                  alpha=0.01)
#classifierRF = RandomForestRegressor(n_estimators=200, max_depth=None,min_samples_split=2, random_state=0,criterion='mse');
#classifierTree = DecisionTreeRegressor(max_depth=500,min_samples_split=2, random_state=0,criterion='mse')


"""
########################## VARIAVEIS GLOBAIS ############################################################################
"""
execult = [];
tamanho = 0;
txErro = [];
real = [];
treinou = False
#classifierInit = list();

classifier = list();
igd = list();

"""
######################################## FUNÇÕES PARA SALVAR OS VALORES ##################################################
"""

def Leitura(name):
	arquivo = open(name,'r')
	linhas = arquivo.readlines()
	retorno = []

	for linha in linhas:
		retorno.append(eval(linha.replace("[","").replace("]","").replce(" ",",")))
	arquivo.close()

	return retorno


def Save(lista, name):
    import numpy as np
#    arquivo = open(name,'w')
#    i = 0
#    while i < len(lista):    
#        arquivo.writelines((lista[i]))
#        i += 1
#    arquivo.close()
    np.savetxt(name, lista)

def LeituraIGD(name):
    arquivo = open(name,'r')
    linhas = arquivo.readlines()
    retorno = []
    
    for linha in linhas:
        retorno.append(float(linha))
    
    return retorno

def SaveIGD(lista, name):
    import numpy as np
    arquivo = open(name,'w')
    i = 0
    media = 0;
    while i < len(lista):
        media += lista[i]
        arquivo.write(str(lista[i])+'\n')
        i += 1
    media = media/len(lista)
    std = np.std(np.asarray(lista));
    arquivo.write(str(media)+ "+/-" + str(std) + '\n')
    arquivo.close()


def EscreveArquivo(indice, lista, name):
    arquivo = open(name,'w')
    i = 0
    while i < len(lista):    
        arquivo.write(str(indice[i]) +" "+ str(lista[i])+'\n')
        i += 1
    arquivo.close()


def EscreveArquivoC(indice, compara ,lista, name):
    arquivo = open(name,'w')
    i = 0
    arquivo.write("Indice" +" "+ "Predição"+" "+"Rotulo"+" "+"Erro"+'\n')
    while i < len(lista):
        arquivo.write(str(indice[i]) +" "+str(compara[i])+" "+ str(lista[i])+'\n')
        i += 1
    arquivo.close()




@app.route("/classificador", methods=['GET', 'POST'])
def classificador():
    global classifier
    global txErro
    global real 
    global classifier_name
    global population_size
    global not_first_run
    global classifier_num_of_epochs
    #global classifierInit
    
    try:
        not_first_run
    except NameError:
        archive_old_mse()
        not_first_run = True

    classifier_num_of_epochs = None

    classifierInit = list()
    message = request.get_json(silent=True)
    #nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    #nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0

    # Message id doesn't make sense because it's the same
    # object used in other cases.
    classifier_name = message['algoritmo']
    tagProblem = message['processar']
    nObj = np.asarray(message['objetivos'])
    if nObj[0] == 3:
        population_size = 92
    else:
        population_size = 764
    
    # print("classificador:", classifier_name)
    if classifier_name.startswith("SVM"):
        classifierInit = SVR(kernel='rbf', C=1e3, gamma=0.1, tol=0.01)
    elif classifier_name.startswith("TREE"):
        for i in range(nObj[0]):
            classifierInit.append(DecisionTreeRegressor(max_depth=500,min_samples_split=2, random_state=0,criterion='mse'))    
    elif classifier_name.startswith("RAMDOMFOREST"):
        for i in range(nObj[0]):
            classifierInit.append(RandomForestRegressor(n_estimators=200, max_depth=None,min_samples_split=2, random_state=0,criterion='mse'))  
    elif classifier_name.startswith("MLP"):
        classifierInit = MLPRegressor(hidden_layer_sizes=(10,),max_iter=1000)
    elif classifier_name.startswith("NO-SURROGATE") or classifier_name.startswith("RANDOM"):
        classifierInit = None
    elif classifier_name.startswith("LSTM") or classifier_name.startswith("RNN"):
        # Clear any session, in case it's not the first one.
        keras.backend.clear_session()

        # Clear GPU memory
        cuda.select_device(0)
        cuda.close()

        # Clear classifiers if they exist
        if 'classifier' in globals():
            del classifier
        gc.collect()

        classifier_num_of_epochs = int(classifier_name.split('_')[2])

        if tagProblem.startswith('WFG'):
            if population_size == 92:
                dim_2 = 14
            else:
                dim_2 = 19
        elif tagProblem.startswith('DTLZ'):
            if population_size == 92:
                dim_2 = 12
            else:
                dim_2 = 19
        else:
            # Invalid dimension
            dim_2 = -1
        for i in range(nObj[0]):
            # print('Modeling NN classifier[' + str(i) + '].')
            classifierInit.append(NeuralNetwork(population_size, dim_2, population_size, classifier_name))
    classifier = classifierInit
    # print("-------------- STARTING FIT ----------------")
    # with open("nSolution.txt", "r") as fp:
    #     nSolution = np.array(json.load(fp))
    # with open("nObj.txt", "r") as fp:
    #     nObj = np.array(json.load(fp)).transpose()
    # # print("INPUT SHAPE =", np.array([nSolution]).shape)
    # for i in range(len(nObj)):
    #     classifier[i].fit(np.array([nSolution]), np.array([nObj[i]]), 1, 0, False)
    # print("-------------- END STARTING FIT ----------------")
    
    return json.dumps({"retorno": []})

"""
##################################################### TREINAMENTO ########################################################
"""
@app.route("/treinamento", methods=['GET', 'POST'])
def treino():
    global classifier
    global txErro
    global real
    global tamanho
    global execult
    global treinou
    global population_size
    global classifier_num_of_epochs
    
    lista = []
    message = request.get_json(silent=True)
    # with open("nSolution.txt", "w") as fp:
    #     json.dump(message["solucoes"], fp)
    # with open("nObj.txt", "w") as fp:
    #     json.dump(message["objetivos"], fp)
    nSolution = np.asarray(message["solucoes"])
    nObj = np.asarray(message["objetivos"])
    #print(f'O tamanho do treino: {len(nObj)}')
    #processar = message["processar"]
    #erro = message["erro"]  
    
    nObj = nObj.transpose()
    # print  ("TREINA", "nSolution", nSolution)
    # print  ("TREINA", "nObj", nObj)
    with open ("out.txt", "a") as f:
        f.write ("TREINA\n")

    # Fit for LSTM
    if classifier != None and isinstance(classifier[0], NeuralNetwork):
        if len(nSolution) > population_size:
            nSolution = np.array(np.split(nSolution, len(nSolution)//population_size))
        else:
            nSolution = np.array([nSolution])
        for i in range(len(nObj)):
            # print('Fitting NN classifier[' + str(i) + '].')
            if len(nObj[i] > population_size):
                obj_now = np.array(np.split(nObj[i], len(nObj[i])//population_size))
            else:
                obj_now = np.array([nObj[i]])
            
            classifier[i].fit(nSolution, obj_now, classifier_num_of_epochs, 0, False)

    # For for anything else
    elif classifier != None:
        for i in range(len(nObj)):
            classifier[i] = classifier[i].fit(nSolution, nObj[i])
    #treinou = True
    
    #versão Batch
    #classifier = classifier.fit(nSolution, nObj);
    #y_predict = classifier.predict(nSolution)
    
    #i = 0
    #for valor in y_predict:
    #    real.append(nObj[i])
    #    txErro.append(y_predict[i])
    #    i+=1
    
    return json.dumps({"retorno": lista})




"""
######################################################### SALVAR #########################################################
"""
@app.route("/save", methods=['GET', 'POST'])
def save():
    global classifier
    global txErro
    global real
    global tamanho
    global execult
    global treinou
    global classifierInit
    
    message = request.get_json(silent=True)
    nome = message["algoritmo"]
    nSolution = np.asarray(message["solucoes"])
    nObj = np.asarray(message["objetivos"])
    #salva o erro e o classificador
    #joblib.dump(classifier, "TREE.pkl")
    if real:    
        SaveIGD(nSolution.tolist(), nome+".txt")
    
    
    Save(nSolution.tolist(), nome+".txt")
    Save(nObj.tolist(), nome+"_Objetivos.txt")
    Save(np.zeros((len(nObj), 1)).tolist(), nome+"_Cons.txt")
#    if txErro:
#        Save(txErro,"DadosPreditos_"+nome+".txt")
#    
#    real = []
#    txErro = []
#    treinou = False
#    classifier = classifierInit
    
    return json.dumps({"retorno": []})

"""
#################################### CLASSIFICA AS SOLUCOES ###############################################################
"""

@app.route("/classifica", methods=['GET', 'POST'])
def classifica():
    global classifier
    global txErro
    global real
    global classifier_name
    
    message = request.get_json(silent=True)
    nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0
    
    #classifier = joblib.load("TREE30k.pkl") 
    y_predict = list()

    nObj = nObj.transpose()
    if classifier != None:
        if isinstance(classifier[0], NeuralNetwork):
            for i in range(len(nObj)):
                # Classifier can be a keras Sequential model, in that case, could have a variable
                # batch_size.
                y_predict.append(classifier[i].predict(np.array([nSolution]))[0])
        else:
            for i in range(len(nObj)):
                y_predict.append(classifier[i].predict(nSolution))
                # print ("score classifier", i, classifier[i].score(nSolution, nObj)) TODO: Discover why it's NOT working!!!

    # Read expected objective from file
    with open ("../jMetal-master/jmetal-exec/out.txt", 'r') as f:
        s = f.read()
        obj_expected = json.loads(s)

    if classifier == None:
        # If surrogate is random, instead of returning expected objetive,
        # it will be randomly generated.
        if classifier_name.startswith('RANDOM'):
            obj_random = []
            for i in range(len(obj_expected)):
                obj_random.append([random.random() for x in range(3)])
            evalSurrogate(obj_expected, obj_random)
            obj_expected = obj_random
        y_predict = [[] for i in range(len(obj_expected[0]))]
        for indv in obj_expected:
            for idObj in range(len(indv)):
                y_predict[idObj].append(indv[idObj])
    else:
        obj_predict = []
        for i in range(len(y_predict[0])):
            obj_predict.append ([])
        for i in range(len(y_predict[0])):
            for j in range(len(y_predict)):
                obj_predict[i].append(y_predict[j][i])

        evalSurrogate(obj_expected, obj_predict)

    return json.dumps({"retorno": np.asarray(y_predict).tolist()})

def archive_old_mse():
    old_mse = []
    for f in os.listdir('.'):
        if f.startswith('mse_') and '.txt' in f:
            old_mse.append(os.path.abspath(f))
    if len(old_mse):
        folder_name = 'mse_backp_' + str(int(time.time()))
        os.makedirs(folder_name)
        folder_name = os.path.abspath(folder_name)
        for f in old_mse:
            subprocess.Popen(['mv', f, folder_name])

def evalSurrogate(obj_expected, obj_predict):
    mse_out_fname = "mse_" + classifier_name + ".txt"
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from pprint import pprint
    from statistics import mean
    obj_expected = np.array([np.array(xi) for xi in obj_expected])
    obj_predict = np.array([np.array(xi) for xi in obj_predict])
    mae = mean_absolute_error(obj_expected, obj_predict)
    
    # Inicialmente, fazendo media de todos os valores avaliados.
    mse = mean_squared_error(obj_expected, obj_predict)
    with open(mse_out_fname, "a") as f:
        f.write(str(mse) + '\n')
    # print("mse:", mse)

    """
    print ("obj_predict:")
    pprint (list([list (x)for x in obj_predict]))
    print ("obj_expected:")
    pprint (list([list (x)for x in obj_expected]))
    print ("diff:")
    pprint (list([list (x)for x in obj_predict - obj_expected]))
    """

    """
    # Testing mean_square_error
    ## 1D
    diff0 = obj_predict[0] - obj_expected[0]
    print ("diff0 = ", obj_predict[0] - obj_expected[0])
    print ("diff0**2 = ", [x**2 for x in diff0])
    print ("average of diff0**2 = ", mean ([x**2 for x in diff0]))
    print ("sqrt of sum of diff0**2 = ", mean ([x**2 for x in diff0])**0.5)
    print ("MSE indv 0:", mean_squared_error (obj_predict[0], obj_expected[0]))
    # print ("MSE indv 1 obj 1:", mean_squared_error (obj_predict[0][0], obj_expected[0][0]))
    ## 2D:
    sumSq = 0
    for i in range (len (obj_expected)):
        for j in range (len (obj_expected[i])):
            sumSq += (obj_expected[i][j] - obj_predict[i][j])**2
    avrSq = sumSq / (len(obj_expected) * len(obj_expected[0]))
    print ("avrSq = ", avrSq)
    print ("MSE = ", mse)
    """

"""
############################## INICIALIZA AS VARIAVEIS COM A FUNCAO LHS ####################################################
"""
@app.route("/inicializa", methods=['GET', 'POST'])
def Inicializa():
    
    message = request.get_json(silent=True)
    
    nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0
    # print(f'O tamanho do treino: {nObj, nSolution}')
    retorno = lhs(nObj[0], samples=nSolution[0])
    retorno = np.asarray(retorno)


    return json.dumps({"retorno": retorno.tolist()})



@app.route("/saveIGD", methods=['GET', 'POST'])
def saveIGD():
    global igd
    message = request.get_json(silent=True)
    nome = message["algoritmo"]    
    
    SaveIGD(igd, nome+".txt")
    
    igd = list()
    
    return json.dumps({"retorno": []})

@app.route("/IGD", methods=['GET', 'POST'])
def IGD():
    global igd
    
    message = request.get_json(silent=True)
    nObj = np.asarray(message["objetivos"])
    
    igd.append(nObj[0])
    
    return json.dumps({"retorno": []})

@app.route("/guarda", methods=['GET', 'POST'])
def Base():
    global real

    
    message = request.get_json(silent=True)
    
    nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0
    
    i = 0;
    for valor in nObj:
        real.append(nObj[i])
        i+=1
    

    return json.dumps({"retorno": []})


