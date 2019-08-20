#!/usr/bin/env python
# coding: utf-8

# # Ayudantía 2
# 
# **Magister en Finanzas** 
# 
# **Autor**: *Gabriel Cabrera  G.*
# 
# **Fecha**: 21 de Agosto del 2019

# # Generando Datos
# 
# ## Series

# 1. Utilizando NumPy genere un Pandas con estructura Series con 10 datos aleatorios.

import pandas as pd
import numpy as np

pandas_series_a = pd.Series(np.random.rand(10)) 
pandas_series_a

# 2. Cambie el *index* de la Series por letras.

pandas_series_a.index = list('abcdefghij')
pandas_series_a

# 3. Agregue dos nuevos números con su respectivo *index*.

pandas_series_a['k'] = float(np.random.rand(1)) 
pandas_series_a['l'] = float(np.random.rand(1))

pandas_series_a.name = 'series_a'
pandas_series_a

# 4. Genere una Series a partir del siguiente diccionario: 
# 
#     `{'Santiago': 404495, 'Providencia': 142079, 'Huechuraba': 98671, 'Quilicura': 210410}`

dict_A = pd.Series({'Santiago': 404495, 'Providencia': 142079, 'Huechuraba': 98671, 'Quilicura': 210410})
dict_A

# 5. Genere otra series con el diccionario anterior pero que el index sea:
# *Santiago*, *Providencia*, *Huechuraba*, *San Miguel*. ¿Que se observa?
# 

dict_B = pd.Series({'Santiago': 404495, 'Providencia': 142079, 'Huechuraba': 98671, 'Quilicura': 210410}, 
                   index = ['Santiago', 'Providencia', 'Huechuraba', 'San Miguel'])
dict_B


# ## DataFrame
# 
# 1. Utilizando Pandas genere los siguientes DataFrames:

df_a = pd.DataFrame(np.arange(12.).reshape(3, 4), columns=list('abcd'))
df_a

df_b = pd.DataFrame(np.arange(20.).reshape(4, 5), columns=list('abcde'))
df_b

# 2. Para cada DataFrame creado en (1):
# 
#     a. Seleccione la primera fila de cada columna.

df_a.loc[0]

df_a.iloc[0]

df_b.loc[0]

df_b.iloc[0]

#     b. Seleccione la columna c y d. 

df_a.loc[:,['c', 'd']]

df_a.iloc[:,2:4]

df_b.loc[:,['c', 'd']]

df_b.iloc[:,2:5]

#     c. Seleccione la columna a y b, luego filtre los valores menores a 5 de la columna a.

df_a.loc[:,['a', 'b']][df_a.a <= 5]

# 3. Sume los dos DataFrames.

df_c = df_a + df_b
df_c

# 4. A partir de la pregunta anterior, reemplace por cero aquellos valores con `NaN`.

df_c.fillna(0)

# 5. Utilizando NumPy genere un DataFrame que contenga 4 filas y 3 columnas, los datos deben ser aleatorios y aceptar negativos.
# 
#     a. Obtenga el valor absoluto de cada observación.

DF_A = pd.DataFrame(np.random.randn(12).reshape(4, 3))
DF_A

abs(DF_A)

#     b. Utilizando una función anónima calcule el promedio de cada columna.

DF_A.apply(lambda x: x.mean(), axis = 0) # retorna la columna
DF_A.apply(lambda x: x.mean(), axis = 1) # retorna la fila

#     c. Construya una función que permite calcular el promedio, el valor mínimo y máximo de cada columna. 

def f_mean_min_max(x):
    return pd.Series([x.mean(), x.min(), x.max()], index=['mean', 'min', 'max'])

DF_A.apply(f_mean_min_max, axis = 0)

# # Manipulación de Datos
# 
# 1. Importe la base de datos **credits.csv**.

credit_pd = pd.read_csv("credit.csv")
credit_pd

# 2. Realice la estadística descriptiva.

credit_pd.describe()

# 3. Seleccione aquellas observaciones que sean del género ('Gender') femenino ('Female').

credit_pd[credit_pd['Gender'] == 'Female'].head(5)

# 4. Muestre los individuos que: 
# 
#     a. Posean una renta mensual mayor a 1000.

credit_pd[credit_pd['Mo_Income'] > 1000].head(5)

#     b. Posean una renta mensual mayor a 1000 y que sean del género femenino.

credit_pd[(credit_pd['Mo_Income'] > 1000) & (credit_pd['Gender'] == 'Female')].head(5)

#     c. Posean una renta mensual mayor a 1000 o que sean del género femenino.

credit_pd[(credit_pd['Mo_Income'] > 1000) | (credit_pd['Gender'] == 'Female')].head(5)

#     d. Ordene los datos de mayor a menor según ingresos, muestre las 10 primeras observaciones. 

credit_pd.sort_values(by='Mo_Income', ascending=False).head(10)

