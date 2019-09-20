# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import pandas as pd 
import pandas_datareader.data as web
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
from tictoc import tic, toc, toc2
import sys

#%%
raw_data_a = {
        'id': ['1', '2', '3', '4', '5'],
        'nombre': ['Tom', 'Will', 'Tom', 'Jennifer', 'Charlize'], 
        'apellido': ['Cruise', 'Smith', 'Hanks', 'Aniston', 'Theron']}
df_a = pd.DataFrame(raw_data_a, columns = ['id', 'nombre', 'apellido'])
df_a

raw_data_b = {
        'id': ['4', '5', '6', '7', '8'],
        'nombre': ['Julia', 'Nicole', 'Emma', 'George', 'Al'], 
        'apellido': ['Roberts', 'Kidman', 'Watson', 'Clooney', 'Pacino']}
df_b = pd.DataFrame(raw_data_b, columns = ['id', 'nombre', 'apellido'])
df_b

raw_data_c = {
        'id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data_c, columns = ['id','test_id'])
df_n

df_nueva = pd.concat([df_a, df_b], axis=0) # .reset_index().drop(columns=['index'])
pd.concat([df_a, df_b], axis=1)

# por variable
pd.merge(df_nueva, df_n, on='id')
pd.merge(df_nueva, df_n, left_on='id', right_on='id')

# intercción y unión
pd.merge(df_a, df_b, on='id', how='outer')
pd.merge(df_a, df_b, on='id', how='inner')

# por derecha e izquierda
pd.merge(df_a, df_b, on='id', how='right')
pd.merge(df_a, df_b, on='id', how='left')

# agregar nombre a las columnas repetidas
pd.merge(df_a, df_b, on='id', how='left', suffixes=('_left', '_right'))

# por index
pd.merge(df_a, df_b, right_index=True, left_index=True)

# %%
tickers = ["MSFT", "NVDA", "AMD", "AAPL"]

start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime.today()

tickers_lista = []

for ticker in range(len(tickers)):
    panel_data = web.DataReader(tickers[ticker], "yahoo", start_date, end_date) 
    panel_data["Symbol"] = tickers[ticker]
    tickers_lista.append(panel_data)

stocks_DF = pd.concat(tickers_lista, axis=0) # por filas 
# stocks_DF.to_csv("stocks.csv")
# pd.read_csv("stocks.csv")
stocks_DF = stocks_DF.loc[:,["Close", "Symbol"]]

# %%
msft_DF = tickers_lista[0]
msft_close = msft_DF.Close
msft_lag = msft_DF.Close.shift(1)

msft_returns = np.log(msft_close/msft_lag).dropna()
msft_returns = msft_returns.to_frame()
msft_returns.columns = ['returns']

# merge basado en el index 
msft_merge_outer = pd.merge(msft_DF, msft_returns, left_index=True, right_index=True, how='outer')
msft_merge_inner = pd.merge(msft_DF, msft_returns, left_index=True, right_index=True, how='inner')
msft_merge_left = pd.merge(msft_DF, msft_returns, left_index=True, right_index=True, how='left')
msft_merge_right = pd.merge(msft_DF, msft_returns, left_index=True, right_index=True, how='right')

# %% 
def returns_stocks(x):
    stock_cierre = x
    stock_cierre_lag = stock_cierre.shift(1)
    stock_returns = np.log(stock_cierre/stock_cierre_lag) # no va el dropna()
    return stock_returns

stocks_DF['Return'] = stocks_DF.groupby(['Symbol'])['Close'].apply(returns_stocks)
stocks_DF = stocks_DF.dropna()

returns_DF = stocks_DF.loc[:,["Symbol", "Return"]]
returns_DF['Date'] = returns_DF.index

# %% 
pivot_DF = returns_DF.pivot(index='Date', columns='Symbol', values='Return')

melt_DF = pd.melt(pivot_DF.reset_index(), id_vars='Date', value_vars=tickers)
melt_DF.index = melt_DF['Date']
melt_DF = melt_DF.loc[:,['Symbol', 'value', 'Date']].rename(columns={'value':'Return'})
melt_DF = melt_DF.drop(columns=['Date'])

# %%
cumsum_returns = pivot_DF.cumsum()*100

# Tools > preferences > IPython console > Graphics > Graphics backend > Backend: Automatic
cumsum_returns.plot(color=['cyan', 'magenta', 'green', 'red'], figsize=(15,6))
plt.title("Retornos Acumulados")
plt.xlabel('Fecha')
plt.ylabel('Retorno Acumulado (%)')
plt.legend(title="Tickers")

#%% 
# fibonacci recursive
def fibo_rec(n):
    if n <= 2:
        return 1 
    return fibo_rec(n-1) + fibo_rec(n-2)

complexA = []

for i in range(1, 31):
    tic()
    fibo_value = fibo_rec(i)
    tiempo_final = toc2()
    iteracion = [i, tiempo_final]
    complexA.append(iteracion)

fibo_recursivo = pd.DataFrame(complexA)
fibo_recursivo.columns = ['iter', 'tiempo']

fibo_recursivo.plot(x="iter", y = "tiempo", kind='line')
plt.legend(['Recursivo'])

complexB = []
sizeA = []

for i in range(1, 31):
    fibo_value = fibo_rec(i)
    fibo_size = sys.getsizeof(fibo_value) 
    iteracion = [i, fibo_size]
    complexB.append(iteracion)
    
fibo_comlplexity = pd.DataFrame(complexB)
fibo_comlplexity.columns = ['iter', 'size']

fibo_comlplexity.plot(x="iter", y = "size", kind='line')
plt.legend(['Recursivo'])

#%%
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(seed=777)
resultado_plus_row = []

n_col = 10 
n_row = 10 
n_row_final = 100000000 
n_row_delta = 10000 
n_row_limite =1000000 

while n_row<n_row_final:
    Y = np.random.rand(n_row,1)     
    X = np.random.rand(n_row,n_col) 
    
    reg_lineal = LinearRegression() 
    
    tic()                          
    reg_lineal.fit(X, Y)            
    toc_sklearn_linear_reg = toc2() 
    
    a=[n_row,toc_sklearn_linear_reg]
    resultado_plus_row.append(a)   
    
    n_row=n_row +n_row_delta       
              
    if n_row>n_row_limite:         
        break         

resultado_plus_row_df = pd.DataFrame(resultado_plus_row,columns = ['nrow', 'time'])
graph = resultado_plus_row_df.plot(x='nrow', y='time',kind='line',title='Adding Rows')