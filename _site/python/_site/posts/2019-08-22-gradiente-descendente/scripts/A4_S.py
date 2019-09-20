#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:36:17 2019

@author: gabriel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Gradiente Descendiente"""

np.random.seed(42)

X = 3*np.random.rand(100, 1)
y = 2 + 2*X + np.random.randn(100, 1)

# X_stack = np.vstack((X,np.ones(np.shape(X)))).T#

#%%
X_b = np.c_[np.ones((100, 1)), X]
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta

#%%
X_new = np.array([[0], [3]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta)
y_predict

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "*")
# plt.axis([0, 2, 0, 15])
plt.show()

#%%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_b, y)

lin_reg.intercept_[0], lin_reg.coef_[0][1]

lin_reg.predict(X_new_b)

#%% 
eta = 0.05 # learning rate
n_iterations = 25
theta_history = []
cost_history = []
m = 100
theta_gd = np.random.randn(2,1)

def cal_cost(theta,X,y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/m) * np.sum(np.square(predictions-y))
    return cost

# random initialization
for ite in range(n_iterations):    
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    theta_gd = theta - eta * gradients
    theta_history.append(theta_gd.T)
    cost_history.append(cal_cost(theta_gd, X_b, y))

theta
theta_gd

mse_df = pd.DataFrame(cost_history).reset_index()
mse_df.columns = ['iter', 'mse']
mse_df.plot(x='iter', y='mse', kind='line')

#%%
plt.figure(figsize=(15,20))

for i in np.arange(1, 9):
    num_fig = i*3 + 1 
    X_newb = np.array([[0], [3]])
    
    X_new_bb = np.c_[np.ones((2, 1)), X_newb] # add x0 = 1 to each instance
    Y_pred = X_new_bb.dot(theta_history[i].T)
    
    plt.subplot(4,2,i)
    
    plt.plot(X_newb, Y_pred, "r-")
    plt.plot(X, y, "*")
    title_str = 'After %d iterations: %0.7f X  + %0.7f'%(num_fig, theta_history[i*3][0,0], theta_history[i*3][0,1])
    plt.title(title_str)

#%% 
"""Stochastic Gradient Descent"""

n_epochs = 50
t0, t1 = 5, 50

# learning schedule hyperparameters
def learning_schedule(t):
    return t0 / (t + t1)
 
theta_sgd = np.random.randn(2,1)

# random initialization
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        eta = learning_schedule(epoch * m + i)
        theta_sgd = theta_sgd - eta * gradients

theta_sgd

#%%
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.05, random_state=42)
sgd_reg.fit(X, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_

#%%
"""Mini-batch gradient descent"""

theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta_mbgd = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta_mbgd) - yi)
        eta = learning_schedule(t)
        theta_mbgd = theta_mbgd - eta * gradients
        theta_path_mgd.append(theta_mbgd)

theta_mbgd
#%%
"""Aplicación: Regresión Lineal"""

import pandas as pd
excel_file= 'data_usd_clp.xlsx'
data = pd.read_excel(excel_file)

#%%
data_dif=data.diff()

data_lag1=data_dif.shift(1)
cols_lag1= [col+"_lag1" for col in data.columns]
data_lag1.columns=cols_lag1

data_lag2=data_dif.shift(2)
cols_lag2= [col+"_lag2" for col in data.columns]
data_lag2.columns=cols_lag2

data_lag3=data_dif.shift(3)
cols_lag3= [col+"_lag3" for col in data.columns]
data_lag3.columns=cols_lag3

data_con_lags=pd.concat([data_dif,data_lag1,data_lag2, data_lag3], axis=1).dropna()

#%%
def crealags(base,lag_ini,nrolags):
    data_dif = base.diff()
    for lags in range(lag_ini,  nrolags+1): # Parte del rezago que definamos, no desde 1
        slag=base.shift(lags).copy(True)
        slag.columns=[str(col) + '_lag'+str(lags) for col in base.columns]
        if lags==lag_ini: # Bloque de datos inicial
            rezagos=pd.concat([slag], axis=1).copy(True) # genera primer bloque de datos 
        else: rezagos=pd.concat([data_dif, rezagos, slag], axis=1).copy(True) # genera el resto del bloque de datos        
    return rezagos

data_rezagos = crealags(data_dif,1,3).dropna()

data_rezagos = data_con_lags.dropna()
Date = data.iloc[4:len(data_dif), 0]
data_rezagos.index = Date
data_rezagos = data_rezagos.drop(columns=['Date', 'Date_lag1', 'Date_lag2', 'Date_lag3'])

#%%
y = data_rezagos.iloc[:,0] 
x = data_rezagos.iloc[:,1:-1]

year_corte = '2015-06-19 00:00:00'

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_tr = scaler.fit_transform(x[x.index < year_corte].copy(True))
y_tr = y[y.index < year_corte].copy(True)

x_tst = scaler.transform(x[x.index > year_corte].copy(True))
y_tst = y[y.index > year_corte].copy(True)

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#%%
modelo = SGDRegressor()  # creo el obj regresor SGD vacio
modelo = modelo.fit(x_tr, y_tr) #lo entreno i.e. encuentro los parametros theta optimos para la data

thetas = modelo.coef_ #pido los coeficientes

preds_tr = modelo.predict(x_tr) # creo las predicciones en la muestra de prueba
mae_tr = mean_absolute_error(y_tr,preds_tr)
rmse_tr = (mean_squared_error(y_tr,preds_tr))**0.5
    
preds_tst = modelo.predict(x_tst)
mae_tst = mean_absolute_error(y_tst,preds_tst)
rmse_tst = (mean_squared_error(y_tst,preds_tst))**0.5

#%%
y_tr = pd.DataFrame(y_tr)
preds_tr = pd.DataFrame(preds_tr,index=y_tr.index,columns=y_tr.columns)

data_tr = pd.concat([y_tr,preds_tr],axis=1)
data_tr.columns = ["real","prediccion"]
data_tr.plot()

y_tst = pd.DataFrame(y_tst)
preds_tst = pd.DataFrame(preds_tst,index=y_tst.index,columns=y_tst.columns)

data_tst = pd.concat([y_tst,preds_tst],axis=1)
data_tst.columns = ["real","prediccion"]
data_tst.plot()

#%%
data_full = pd.concat([data_tr,data_tst])

data_full = pd.concat([data["USDCLP Curncy"],data_full],axis=1)
data_full["pred_nivel"] = data_full["USDCLP Curncy"].shift(1)+data_full["prediccion"]
data_full = pd.concat([data_full["USDCLP Curncy"],data_full["pred_nivel"]],axis=1)
data_full.plot()
