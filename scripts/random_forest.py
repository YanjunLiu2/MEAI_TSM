from __future__ import annotations
import math
import torch
import random
import xlrd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import matplotlib as mpl
import xlwt
import time
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union


from read_data import generate_data




group=[['PbFCl','ZrSiS-UP2','ZrSiTe','AmTe2-x','PrOI','Cu2Sb'],['ZrCuSiAs-HfCuSi2','LaZn0.5Sb2'],['CaBe2Ge2']]
X1,X0 = generate_data(group)
buffer1 = torch.tensor(X1[0]+X1[1], dtype=torch.float64)
buffer0 = torch.tensor(X0[0]+X0[1], dtype=torch.float64)
X1=np.array(buffer1)
X0=np.array(buffer0)
y1=np.array([1 for i in range(len(X1))])
y0=np.array([0 for i in range(len(X0))])

# Step 1: 合并数据
X = np.vstack((X1, X0))        # shape = (861, 12)
y = np.hstack((y1, y0))        # shape = (861, )

# Step 2: Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,               # 例如20%的测试集
    stratify=y,                 # 让每个类按比例出现在train/test中
    random_state=11
)

rf = RandomForestClassifier(random_state=1)

param_grid = {
    'n_estimators': [100, 200, 500],          # 决策树数量
    'max_depth': [None, 10, 20, 30],          # 每棵树最大深度
    'min_samples_split': [2, 5, 10],          # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 2, 4],            # 叶子节点最小样本数
    'max_features': ['sqrt', 'log2', None]   # 每个树节点考虑的特征数
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                    # 5折交叉验证
    scoring='accuracy',            # 可以选 'accuracy', 'roc_auc', 'f1' 等
    verbose=2,
    n_jobs=-1                # 并行处理
)

# Step 4: 训练
grid_search.fit(X_train, y_train)

# Step 5: 输出最佳参数和测试结果
print("Best parameters found:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Overall accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

import matplotlib.pyplot as plt

importances = best_model.feature_importances_

component = np.abs(importances)  # num corresponds to the specified principal component
    
# Plot setup
plt.figure(figsize=(10, 3))
plt.rcParams['font.family'] = 'Times New Roman'

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Set y-axis limits and ticks
plt.ylim(0, 1)
plt.yticks([0,0.25,0.5,0.75,1])
plt.tick_params(axis='y', length=8, width=2, labelsize=22)
plt.tick_params(axis='x', length=8, width=2, labelsize=30)

# Plot the coefficients as a bar plot
plt.bar(range(len(component)), component, color='steelblue')

# Customize x-ticks with descriptors
plt.xticks(
    range(len(component)),
    ['$\chi_{min}$', '$\chi_{sq}$', '$NE_{max}$', '$NE_{min}$', '$NE_{sq}$', '$NE_{tot}$',
        '$d_{sq}$', '$d_{nn}$', 'fcc', '$EA_{max}$', '$EA_{min}$', '$EA_{sq}$'],
    rotation=60, size=30
)

# Set the x-axis label
plt.ylabel('Importance', size=28, labelpad=10)
# Optionally save the figure
plt.savefig('rf_feature_importance.svg', bbox_inches='tight', transparent=True)


