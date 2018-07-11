# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:21:42 2018

@author: Helio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
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

#Feature selection
clf = ExtraTreesClassifier()
clf = clf.fit(x, y)

model = SelectFromModel(clf, prefit=True)
x_new = model.transform(x)

x_train_cdt, x_test_cdt, y_train_cdt, y_test_cdt = train_test_split(x_new, y, test_size=0.3, random_state=1)

cdt = DecisionTreeClassifier(random_state=1986, criterion='entropy', max_depth=3)

#Treinamento com o teste retirado da base de treinamento
cdt.fit(x_train_cdt, y_train_cdt)
saidas_cdt = cdt.predict(x_test_cdt)

#Treinamento com a base inteira
cdt.fit(x, y)

from sklearn.model_selection import cross_val_score

score_dt = cross_val_score(cdt, x_train_cdt, y_train_cdt,
                            scoring='accuracy', cv=5)

print(score_dt.mean())


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test_cdt[:], saidas_cdt[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_cdt.ravel(), saidas_cdt.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Decision Tree')
plt.legend(loc="lower right")
plt.show()

fluxos_teste4 = pd.read_csv('test4-2.csv',usecols=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
fluxos_teste4_y = pd.read_csv('test4-2.csv',usecols=[45])
fluxos_teste4_y = np.array(fluxos_teste4_y)
fluxos_tst_scaled4 = sc.fit_transform(fluxos_teste4)
saidas4 = cdt.predict(fluxos_tst_scaled4)
print('Score Teste4: ', cdt.score(fluxos_tst_scaled4, fluxos_teste4_y))

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
plt.title('Decision Tree')
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
saidas2 = cdt.predict(fluxos_tst_scaled2)
print('Score Teste2: ', cdt.score(fluxos_tst_scaled2, fluxos_teste2_y))

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
plt.title('Decision Tree')
plt.legend(loc="lower right")
plt.show()
