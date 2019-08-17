#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:58:06 2019

@author: gabriel
"""

'''
 condicionales, loops y otros control flow
'''

''' a '''
# forma 1 
seq_a = [1, 2, None, 4, 5, 6, None] 
valor = 0

for i in range(len(seq_a)):
    if seq_a[i] != None:
        valor += seq_a[i]
 
# forma 2
seq_a = [1, 2, None, 4, 5, 6, None] 
valor = 0
        
for i in seq_a:
    if i is None:
        continue
    valor += i 

''' b '''

# forma 1    
seq_b = list(range(1, 8, 1))
total_hasta_5 = 0
   
for i in seq_b:
    if i == 5:
        break
    total_hasta_5 += i

total_hasta_5

''' c '''

suma = 0

for i in range(100000):
    if i % 3 == 0 or i % 5 == 0:
        suma += 1

''' 
funciones 
'''

'''a'''

def elevado(n):
    numero = list(range(1, n+1))
    suma = 0
    for i in numero:
        suma += i**2
    return suma    

elevado(1)

'''b'''

def divisible(n):
    if n % 2 is 0:
        print("Es dividible")
    else:
        print("No es divisible")

divisible(2)

'''c'''

def media_aritmetica(x):
    return sum(x)/len(x)
    
datos = [1, 2, 3, 4, 5, 6]
media_aritmetica(datos)

''' 
introduccion a numpy
'''

''' 
basicos
'''

import numpy as np
from numpy.linalg import inv

# 1 
A = np.array([3, 0, 2, 2, 0, 2, 0, 1, 1]).reshape(3,3)

# 2
A.T
np.transpose(A)

# 3
inv(A)

# 4.a

B = np.array([2, 4, 5, -6]).reshape(2, 2)
C = np.array([9, -3, 3, 6]).reshape(2, 2)

B + C

# 4.b
np.dot(B, C)

''' 
intermedios
'''

# 1 
lista_vacia = []

for i in range(3):
     lista_vacia.append(A[i,2])

# 2.a
a = np.array([1, 1, 1, 3, -2, 1, 2, 1, -1]).reshape(3,3)
b = np.array([6, 2, 1]).reshape(-1, 1)

print(np.linalg.solve(a,b))

# 2.b
c = np.array([3, 4, -5, 1, 2, 2, 2, -1, 1, -1, 5, -5, 5, 0, 0, 1]).reshape(4,4)
d = np.array([10, 5, 7, 4]).reshape(-1, 1)
 
print(np.linalg.solve(c,d))

''' 
avanzados
'''

import pandas as pd

# 1
advertising = pd.read_csv('Advertising.csv')

# 2
# Utilizando solo numpy creamos la regresion sales ~ TV 
n = np.array(advertising.sales).size
y = np.array(advertising.sales).reshape(n, 1)
x = np.array(advertising.TV).reshape(n, 1)
X = np.append(np.ones([n, 1]), x, axis=1).reshape(n, 2)

X_inv = np.linalg.inv(np.dot(np.transpose(X),X))
X_trans_y = np.dot(np.transpose(X),y)
betas = np.dot(X_inv,X_trans_y)

# constate b0 
np.around(betas[0], 3)

# coeficiente b1
np.around(betas[1], 3)
