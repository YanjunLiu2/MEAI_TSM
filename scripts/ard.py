
from __future__ import annotations
import math
import torch
import gpytorch
import random
import xlrd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import matplotlib as mpl
import xlwt
import time
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from MEAI_TSM.GP import make_and_fit_classifier_ard
from MEAI_TSM.GP import compute_accuracy
from MEAI_TSM.GP import cluster_lengthscales
from MEAI_TSM.GP import ard_lengthscales
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from MEAI_TSM.read_data import generate_data
from MEAI_TSM.GP import rescale



#device = torch.device('cuda:0')
device = torch.device('cpu')
detype = torch.double
print(torch.__version__)
print('device', torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name(device))
nfeature=12
rank=2
seed=17
front='17r2ae-2'
group=[['PbFCl','ZrSiS-UP2','ZrSiTe','AmTe2-x','PrOI','Cu2Sb'],['ZrCuSiAs-HfCuSi2','LaZn0.5Sb2'],['CaBe2Ge2']]
X1,X0 = generate_data(group)
buffer1 = torch.tensor(X1[0]+X1[1], dtype=torch.float64)
buffer0 = torch.tensor(X0[0]+X0[1], dtype=torch.float64)
print('totallen',len(buffer1)+len(buffer0))
buffer1 = buffer1.to(device)
buffer0 = buffer0.to(device)
acc_list, acc_list2, state_dict_list, mll_list, tmll_list, acc0_list, acc1_list, mean_list,tmll2_list = [], [], [], [], [], [], [], [], []

ardmatrix = torch.zeros([nfeature, nfeature], dtype=torch.float)
ardmatrix = ardmatrix.to(device)


for i in range(6):
    print("trying seed: ", i)
    torch.random.manual_seed(i+seed)

    # note that now the training set is not fixed
    # lets use 80%
    shuffled_inds0 = torch.randperm(buffer0.shape[0])
    shuffled_inds1 = torch.randperm(buffer1.shape[0])
    #lout0 = int(float(len(shuffled_inds0)) * 0.9)
    #lout1 = int(float(len(shuffled_inds1)) * 0.9)
    #shuffled_inds0 = shuffled_inds0[:lout0]
    #shuffled_inds1 = shuffled_inds1[:lout1]
    cv1 = int(float(len(shuffled_inds1))*0.2)
    cv0 = int(float(len(shuffled_inds0))*0.2)
    split0=[]
    split1=[]
    for j in range(5):
        split0.append(shuffled_inds0[:cv0])
        shuffled_inds0=shuffled_inds0[cv0:]
        split1.append(shuffled_inds1[:cv1])
        shuffled_inds1=shuffled_inds1[cv1:]
    for j in range(5):
        copy0=copy.deepcopy(split0)
        testset0=copy0[j]
        del(copy0[j])
        copy1=copy.deepcopy(split1)
        testset1=copy1[j]
        del(copy1[j])
        trainset0=torch.cat((copy0[0],copy0[1],copy0[2],copy0[3]))
        trainset1=torch.cat((copy1[0],copy1[1],copy1[2],copy1[3]))
        train_x = torch.cat((buffer0[trainset0], buffer1[trainset1]), 0)
        train_y = []
        for tt in range(int(cv0*4.)):
            train_y.append(0)
        for tt in range(int(cv1*4.)):
            train_y.append(1)
        train_y = torch.tensor(train_y)

        test_x = torch.cat((buffer0[testset0], buffer1[testset1]), 0)
        test_y = []
        for tt in range(cv0):
            test_y.append(0)
        for tt in range(cv1):
            test_y.append(1)
        test_y = torch.tensor(test_y)
        print(train_x.shape, train_y.shape)
        with gpytorch.settings.max_cholesky_size(2000):

            model1, mll1 = make_and_fit_classifier_ard(train_x, train_y, rank=rank,
                                                   lr=0.05)
            testacc, trainacc, acc0, acc1 = compute_accuracy(model1, model1.likelihood, test_x, test_y,
                                                                          train_x, train_y)
            ard_matrix, mmean = ard_lengthscales(model1)
        print('means:',mmean)
        
        ardmatrix = ardmatrix + ard_matrix.detach()
        mean_list.append(mmean)
        acc_list.append(testacc)
        acc_list2.append(trainacc)
        acc0_list.append(acc0)
        acc1_list.append(acc1)

        mll_list.append(mll1)
        state_dict_list.append(model1.state_dict)

        model1.eval()
        likelihood1 = DirichletClassificationLikelihood(test_y, learn_additional_noise=True)
        likelihood1 = likelihood1.to(device)
        model1 = model1.to(device)
        likelihood1.eval()
        test_x = test_x.to(device)
        output = model1(test_x)
        tmll = ExactMarginalLogLikelihood(likelihood1, model1)
        tmll = tmll.to(device)
        print('a', model1.covar_warp.outputscale)
        #print(-tmll(output, likelihood1.transformed_targets.to(device)).sum())
        tmll_list.append(-tmll(output, likelihood1.transformed_targets.to(device)).sum().cpu().detach().numpy())
        #val_y=likelihood1.transformed_targets.to(device)
        #prediction = model1(test_x)
        #tmll2=Normal(prediction.mean[0], torch.diag(prediction.variance[0]) ** 0.5).log_prob(val_y[0])+Normal(prediction.mean[1], torch.diag(prediction.variance[1]) ** 0.5).log_prob(val_y[1])
        #tmll2_list.append(tmll2.cpu().detach().numpy())
        #print(val_y.shape)
        #print(-tmll(output, likelihood1.transformed_targets.to(device)).sum(),-tmll2.cpu().detach().numpy())


ardmatrix1 = ardmatrix.cpu()
print(ardmatrix1.shape)
ardplot=ardmatrix1.data.div(30.).numpy()
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot()
f = plt.imshow(ardplot, cmap=mpl.cm.BrBG, vmax=np.max(np.abs(ardplot)), vmin=-np.max(np.abs(ardplot)))
plt.rcParams['font.family']='Times New Roman'
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.gca().invert_yaxis()
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
"""plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
           ['$\chi_{min}$', '$\chi_{sq}$', '$d_{sq}$', '$d_{nn}$', 'fcc', '$EA_{max}$', '$EA_{min}$', '$EA_{sq}$'],
           rotation=60,size=13)#'$NE_{max}$'
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7],
           ['$\chi_{min}$', '$\chi_{sq}$', '$d_{sq}$', '$d_{nn}$', 'fcc', '$EA_{max}$', '$EA_{min}$', '$EA_{sq}$'],
           rotation=0,size=13)"""
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           ['$\chi_{min}$', '$\chi_{sq}$', '$NE_{max}$', '$NE_{min}$', '$NE_{sq}$', '$NE_{tot}$', '$d_{sq}$', '$d_{nn}$', 'fcc', '$EA_{max}$', '$EA_{min}$', '$EA_{sq}$'],
           rotation=60,size=13)#'$NE_{max}$'
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           ['$\chi_{min}$', '$\chi_{sq}$', '$NE_{max}$', '$NE_{min}$', '$NE_{sq}$', '$NE_{tot}$', '$d_{sq}$', '$d_{nn}$', 'fcc', '$EA_{max}$', '$EA_{min}$', '$EA_{sq}$'],
           rotation=0,size=13)
plt.tick_params(length=10, width=2, labelsize=13)
#plt.title("Correlation Matrix", fontsize=20)
cb=plt.colorbar(fraction=0.045, format='%.1f')
maxard=np.max(np.abs(ardplot))
cb.set_ticks([-maxard,-maxard/2., 0, maxard/2. ,maxard])
cb.ax.tick_params(labelsize=14)
cb.outline.set_linewidth(1.5)
plt.savefig(front+str(nfeature) + "ARD.svg", bbox_inches="tight", transparent='true')



workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet1')
for i in range(len(mll_list)):
    worksheet.write(0, i, label=mll_list[i])
    worksheet.write(2, i, label=tmll_list[i])
    worksheet.write(4, i, label=acc_list[i])
    worksheet.write(6, i, label=acc_list2[i])
worksheet.write(1, 0, label=np.mean(mll_list))
#worksheet.write(1, 2, label=np.std(tmll2_list))
worksheet.write(3, 0, label=np.mean(tmll_list))
worksheet.write(3, 2, label=np.std(tmll_list))
worksheet.write(5, 0, label=np.mean(acc_list))
worksheet.write(5, 2, label=np.std(acc_list))
worksheet.write(7, 0, label=np.mean(acc_list2))
worksheet.write(7, 2, label=np.std(acc_list2))
worksheet.write(9, 0, label=np.mean(acc0_list))
worksheet.write(9, 2, label=np.mean(acc1_list))

workbook.save(front+str(nfeature) + 'mll_1.xls')


workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet1')
diag = []
order = []
for i in range(len(rawmatrix1)):
    diag.append(rawmatrix1[i][i])
    order.append(i)
for i in range(len(diag)):
    maxx = diag[i]
    place = i
    for j in range(i + 1, len(diag)):
        if abs(diag[j]) > abs(maxx):
            maxx = diag[j]
            place = j
    diag[place] = diag[i]
    diag[i] = maxx
    tt = order[place]
    order[place] = order[i]
    order[i] = tt
for i in range(10):
    worksheet.write(0, i, label=float(order[i]))
workbook.save(front+str(nfeature) + 'diag_1.xls')

scopy=summatrix1.data.div(60.).numpy()
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet1')
for i in range(len(scopy)):
    for j in range(len(scopy)):
        worksheet.write(len(scopy)-i,j, label=scopy[i][j])
workbook.save(front+str(nfeature) + 'Cmatrix.xls')