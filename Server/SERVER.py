#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:24:26 2018

@author: joel
"""

from flask import Flask, request, jsonify

from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV



import numpy as np
#from pyDOE import lhs

import json

app = Flask(__name__)

#variaveis globais


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
    #global classifierInit
    
    classifierInit = list();
    message = request.get_json(silent=True)
    #nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    #nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0
    processar = message["processar"]
    nObj = np.asarray(message["objetivos"])
    
    print("classificador:", processar)
    if processar == "SVM":
        classifierInit = SVR(kernel='rbf', C=1e3, gamma=0.1, tol=0.01)
    elif processar == "TREE":
        for i in range(nObj[0]):
            classifierInit.append(DecisionTreeRegressor(max_depth=500,min_samples_split=2, random_state=0,criterion='mse'))    
    elif processar == "RAMDOMFOREST":
        for i in range(nObj[0]):
            classifierInit.append(RandomForestRegressor(n_estimators=200, max_depth=None,min_samples_split=2, random_state=0,criterion='mse'))  
    elif processar == "MLP":
        classifierInit = MLPRegressor(hidden_layer_sizes=(10,),max_iter=1000)
    elif processar == "NO-SURROGATE":
        classifierInit = None

    classifier = classifierInit
    
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
    
    lista = []
    message = request.get_json(silent=True)
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
    if classifier != None:
        for i in range(len(nObj)):
            classifier[i] = classifier[i].fit(nSolution, nObj[i]);
	    
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
    # print("# /SAVE # nome = " + nome)
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
    
    message = request.get_json(silent=True)
    nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0
    
    #classifier = joblib.load("TREE30k.pkl") 
    y_predict = list()

    nObj = nObj.transpose()
    if classifier != None:
        for i in range(len(nObj)):
            y_predict.append(classifier[i].predict(nSolution))
            #print ("score classifier", i, classifier[i].score(nSolution, nObj)) TODO: Discover why it's NOT working!!!

    # Read expected objective from file
    with open ("/home/lclpsoz/Dropbox/Superior/CC-UFS/ICs/3-Andre/proj/jMetal-master/jmetal-exec/out.txt", 'r') as f:
        s = f.read()
        # print ("s: ", type (s), s)
        obj_expected = json.loads(s)
        # print ("lst:", obj_expected)

    # print ("y_predict lens:", len(y_predict), len(y_predict[0]))
    # print ("y_predict types:", type(y_predict), type(y_predict[0]), type(y_predict[0][0]))
    # print ("y_predict:", y_predict)
    if classifier == None:
        y_predict = [[] for i in range(3)]
        for indv in obj_expected:
            assert len(indv) == len(y_predict)
            for idObj in range(len(indv)):
                y_predict[idObj].append(indv[idObj])
    else:
        obj_predict = []
        for i in range(len(y_predict[0])):
            obj_predict.append ([])
        for i in range(len(y_predict[0])):
            for j in range(len(y_predict)):
                obj_predict[i].append(y_predict[j][i])

    # print ("obj_predict:", obj_predict)

    # evalSurrogate(obj_expected, obj_predict)

    #print ("score classifier",0, classifier[0].score(nSolution, obj_expected))

    y_predict = np.asarray(y_predict)
#    i = 0
#    for valor in y_predict:
#        real.append(nObj[i])
#        txErro.append(y_predict[i])
#        i+=1


    return json.dumps({"retorno": y_predict.tolist()})

def evalSurrogate(obj_expected, obj_predict):
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from pprint import pprint
    from statistics import mean
    obj_expected = np.array([np.array(xi) for xi in obj_expected])
    obj_predict = np.array([np.array(xi) for xi in obj_predict])
    mae = mean_absolute_error(obj_expected, obj_predict)
    
    # Inicialmente, fazendo media de todos os valores avaliados.
    mse = mean_squared_error(obj_expected, obj_predict)
    with open ("out.txt", "a") as f:
        f.write (str (mse) + '\n')
    print ("mse:", mse)

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
    print(f'O tamanho do treino: {nObj, nSolution}')
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


