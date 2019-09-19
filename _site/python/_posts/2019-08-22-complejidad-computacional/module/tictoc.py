# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 13:22:28 2017

@author: david
"""

#función tic-toc para calcular tiempos de ejecución
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")
        
def toc2():
    import time
    if 'startTime_for_tictoc' in globals():
        ti=time.time() - startTime_for_tictoc
    else:
        ti= "Toc: start time not set"
    return ti