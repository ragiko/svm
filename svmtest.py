# -*- coding: utf-8 -*-
"""
SVMのTESTコード
参考: http://qiita.com/yasunori/items/8720c85e75b4679cae47
"""
from sklearn.svm import LinearSVC
import numpy as np

# 学習データ
data_training_tmp = np.loadtxt('CodeIQ_auth.txt', delimiter=' ')
data_training = [[x[0], x[1]] for x in data_training_tmp]
label_training = [int(x[2]) for x in data_training_tmp]

# 試験データ
data_test = np.loadtxt('CodeIQ_mycoins.txt', delimiter=' ')

# 学習
estimator = LinearSVC(C=1.0)
estimator.fit(data_training, label_training)

# 予測するよー
label_prediction = estimator.predict(data_test)

print(label_prediction)
