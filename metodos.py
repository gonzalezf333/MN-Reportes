#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import copy as cp


# In[109]:


# Método de la Secante
def secante(c0, c1, epsilon, N_max, funcion):
    vectorc = np.zeros(N_max)
    vectorc[0] = cp.copy(c0)
    vectorc[1] = cp.copy(c1)
    tipo_finalizacion=[0,0,0]
    iteraciones = 0
    for it in range(2, N_max+1, 1):
        vectorc[it] = vectorc[it-1] - funcion(vectorc[it-1])*(vectorc[it-1]-vectorc[it-2])/(funcion(vectorc[it-1])-funcion(vectorc[it-2]))
        #verificamos las condiciones de tolerancia
        err_1 = np.abs(vectorc[it] - vectorc[it-1])
        err_2 = np.abs(funcion(vectorc[it]))
        if err_1 < epsilon:
            tipo_finalizacion[0]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones,tipo_finalizacion
        if err_2 < epsilon:
            tipo_finalizacion[1]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones,tipo_finalizacion
        c = vectorc[it]
        iteraciones += 1
    if iteraciones == N_max:
        tipo_finalizacion[2]=1
    return c, vectorc, iteraciones,tipo_finalizacion


# In[110]:


# Método de Newton-Raphson
def newton_raphson(c0, epsilon, N_max, funcion, fprima):
    vectorc = np.zeros(N_max)
    vectorc[0] = cp.copy(c0)
    iteraciones = 1
    tipo_finalizacion=[0,0,0]
    aux= abs(funcion(vectorc[0]))
    for it in range(1, N_max, 1):
        if fprima(vectorc[it-1])!=0.0:
            vectorc[it] = vectorc[it-1] - (funcion(vectorc[it-1]) / fprima(vectorc[it-1]))
        if fprima(vectorc[it-1])==0.0:
            iteraciones += 1
        #verificamos las condiciones de tolerancia
        err_1 = np.abs(vectorc[it] - vectorc[it-1])
        err_2 = np.abs(funcion(vectorc[it]))
        if err_1 < epsilon:
            tipo_finalizacion[0]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones, tipo_finalizacion
        if err_2 < epsilon:
            tipo_finalizacion[1]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones, tipo_finalizacion
        c = vectorc[it]
        if abs(funcion(c)) < abs(funcion(aux)):
            aux = abs(funcion(c))
        iteraciones += 1
    if iteraciones == N_max:
        tipo_finalizacion[2]=1
    return aux, vectorc, iteraciones, tipo_finalizacion


# In[111]:


# Método de la falsa posición
def falsa_posicion(a, b, epsilon, N_max, funcion):
    vectorc = np.zeros(N_max)
    aa = cp.copy(a)
    bb = cp.copy(b)
    iteraciones = 0
    tipo_finalizacion=[0,0,0]
    for it in range(N_max):
        vectorc[it] = bb - (funcion(bb)*(bb-aa)/(funcion(bb)-funcion(aa)))
        #Verificamos las condiciones de tolerancia
        err_1 = np.abs(vectorc[it] - vectorc[it-1])
        err_2 = np.abs(funcion(vectorc[it]))
        if err_1 < epsilon:
            tipo_finalizacion[0]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones, tipo_finalizacion
        elif err_2 < epsilon:
            tipo_finalizacion[1]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones, tipo_finalizacion
        elif funcion(aa)*funcion(vectorc[it])<0.:
            bb = vectorc[it]
        elif funcion(bb)*funcion(vectorc[it])<0.:
            aa = vectorc[it]
        iteraciones += 1
    c = vectorc(iteraciones)
    if iteraciones == N_max:
        tipo_finalizacion[2]=1
    return c, vectorc, iteraciones, tipo_finalizacion


# In[112]:


# Método de la bisección
def biseccion(a, b, epsilon, N_max, funcion):
    vectorc = np.zeros(N_max)
    aa = cp.copy(a)
    bb = cp.copy(b)
    iteraciones = 0
    tipo_finalizacion=[0,0,0]
    for it in range(N_max):
        c = (aa + bb) / 2
        vectorc[it] = c
        #Verificamos las condiciones de tolerancia
        err_1 = np.abs(vectorc[it] - vectorc[it-1])
        err_2 = np.abs(funcion(vectorc[it]))
        if err_1 < epsilon:
            tipo_finalizacion[0]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones, tipo_finalizacion
        elif err_2 < epsilon:
            tipo_finalizacion[1]=1
            c = vectorc[it]
            iteraciones += 1
            return c, vectorc, iteraciones, tipo_finalizacion
        elif funcion(aa)*funcion(vectorc[it])<0.:
            bb = vectorc[it]
        elif funcion(bb)*funcion(vectorc[it])<0.:
            aa = vectorc[it]
        iteraciones += 1
    c = vectorc(iteraciones)
    if iteraciones == N_max:
        tipo_finalizacion[2]=1
    return c, vectorc, iteraciones, tipo_finalizacion

