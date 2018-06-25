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

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.3, random_state=1)

cdt = DecisionTreeClassifier(random_state=1986, criterion='entropy', max_depth=3)

cdt.fit(x_train, y_train)
saida = cdt.predict(x_test)


from sklearn.model_selection import cross_val_score

score_dt = cross_val_score(cdt, x_train, y_train,
                            scoring='accuracy', cv=5)

print(score_dt.mean())

fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test[:], saida[:])
roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), saida.ravel())
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

