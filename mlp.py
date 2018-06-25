# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:21:42 2018

@author: Helio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

#Carregar Dataset
fluxos = pd.read_csv('dataset_fluxo_bc.csv',usecols=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
fluxos_classes = pd.read_csv('dataset_fluxo_bc.csv',usecols=[45])

#Pre-processamento do dataset

sc = MinMaxScaler(feature_range = (0, 1))
fluxos_scaled = sc.fit_transform(fluxos)


#Convertendo  os datasets em variáveis do numpy
x = np.array(fluxos_scaled)
y = np.array(fluxos_classes)

#Separar conjunto de teste e treinamento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#Treinamento da rede
knn = KNeighborsClassifier(n_neighbors=5)

#Treinamento com o teste retirado da base de treinamento
knn.fit(x_train, y_train)
saidas = knn.predict(x_test)

#print('Score: ', (saidas == y_test).sum() / len(y_test))
#print('Score: ', mlp.score(x_test, y_test))

fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test[:], saidas[:])
roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), saidas.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
