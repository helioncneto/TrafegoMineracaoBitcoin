# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:21:42 2018

@author: Helio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

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

x_train_mlp, x_test_mlp, y_train_mlp, y_test_mlp = train_test_split(x, y, test_size=0.3, random_state=1)

#Treinamento da rede
mlp = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(21,), random_state=1, 
                    learning_rate='constant', learning_rate_init=0.01, max_iter=500, 
                    activation='relu', momentum=0.8, verbose=True, tol=0.000001)

#Treinamento com o teste retirado da base de treinamento
mlp.fit(x_train_mlp, y_train_mlp)
saidas_mlp = mlp.predict(x_test_mlp)

#Treinamento utilizando a base inteira
mlp.fit(x, y)
saidas_mlp = mlp.predict(x_test_mlp)

cm = confusion_matrix(y_test_mlp, saidas_mlp)
print(cm)

tn, fp, fn, tp = confusion_matrix(y_test_mlp, saidas_mlp).ravel()



#print('Acuracia: ', ((tp +tn ) / (tp + fp +tn +fn)))
print('Acuracia: ', mlp.score(x_test_mlp, y_test_mlp))

precision = tp / (tp + fp)
sensibilidade = tp / (tp + fn)
especificidade = tn / (tn + fp)
eficiencia = (sensibilidade + especificidade) / 2

print(precision)
print(sensibilidade)
print(especificidade)
print(eficiencia)

'''
from sklearn.model_selection import cross_val_score

score_mlp = cross_val_score(mlp, x_train_mlp, y_train_mlp,
                            scoring='accuracy', cv=5)

print(score_mlp.mean())
'''



fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test_mlp[:], saidas_mlp[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_mlp.ravel(), saidas_mlp.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Redes Neurais Artificiais')
plt.legend(loc="lower right")
plt.show()

fluxos_teste4 = pd.read_csv('test4-2.csv',usecols=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
fluxos_teste4_y = pd.read_csv('test4-2.csv',usecols=[45])
fluxos_teste4_y = np.array(fluxos_teste4_y)
fluxos_tst_scaled4 = sc.fit_transform(fluxos_teste4)
saidas4 = mlp.predict(fluxos_tst_scaled4)
print('Score Teste4: ', mlp.score(fluxos_tst_scaled4, fluxos_teste4_y))
cm = confusion_matrix(fluxos_teste4_y, saidas4)
print(cm)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(fluxos_teste4_y[:], saidas4[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(fluxos_teste4_y.ravel(), saidas4.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Redes Neurais Artificiais')
plt.legend(loc="lower right")
plt.show()

'''
fluxos_teste3 = pd.read_csv('test3.csv',usecols=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
fluxos_teste3_y = pd.read_csv('test3.csv',usecols=[45])
fluxos_tst_scaled3 = sc.fit_transform(fluxos_teste3)
saidas3 = mlp.predict(fluxos_tst_scaled3)
print('Score Teste3: ', mlp.score(fluxos_tst_scaled3, fluxos_teste3_y))
'''
fluxos_teste2 = pd.read_csv('test2.csv',usecols=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
fluxos_teste2_y = pd.read_csv('test2.csv',usecols=[45])
fluxos_teste2_y = np.array(fluxos_teste2_y)
fluxos_tst_scaled2 = sc.fit_transform(fluxos_teste2)
saidas2 = mlp.predict(fluxos_tst_scaled2)
print('Score Teste2: ', mlp.score(fluxos_tst_scaled2, fluxos_teste2_y))
cm = confusion_matrix(fluxos_teste2_y, saidas2)
print(cm)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(fluxos_teste2_y[:], saidas2[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(fluxos_teste2_y.ravel(), saidas2.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Redes Neurais Artificiais')
plt.legend(loc="lower right")
plt.show()
