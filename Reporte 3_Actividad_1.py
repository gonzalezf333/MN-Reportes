#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Actividad 1
import Funciones_gradiente as FG
from Newton_modificado_multivar import NR_mod_multi
import sympy as sym
import numpy as np


# In[2]:


import pandas as pd


# In[3]:


#sistema de ecuaciones 1
# para hallar una x0,y0 inicial
def f1(x1,x2):
    return np.log((x1)**2+(x2)**2)-np.sin((x1)*(x2))-np.log(2*np.pi)
def f2(x1,x2):
    return np.exp((x1)-(x2)) + np.cos((x1)*(x2))
x = np.linspace(-2,3,100)
y = np.linspace(-2,3,100)
df = pd.DataFrame()
df["x"]=x
df["y"]=y
df["f1(x,y)"]=f1(x,y)
df["f2(x,y)"]=f2(x,y)


# In[4]:


print(df[df['f1(x,y)']<0.001])


# In[5]:


#gradiente descendente
f1 = ['log((x)**2+(y)**2)-sin((x)*(y))-log(2*pi)']
f2 = ['exp(x-y) + cos(x*y)']
f1_cuad = ['(log((x)**2+(y)**2)-sin((x)*(y))-log(2*pi))**2+(exp(x-y) + cos(x*y))**2']
variables_1 = 'x y'
valor_inicial_1 = np.array([[-1.5], [-1.5]]) #aprox fila 9 de df
aprox_inicial_f1,iter1_gd = FG.sec_gradiente(valor_inicial_1, f1_cuad, variables_1, 50,0.05)
print("sistema de ecuaciones 1:")
print("gradiente descendente:")
print("aprox_inicial_f1: ", aprox_inicial_f1)
print("iteraciones:",iter1_gd)


# In[6]:


#newton modificado
def F1(x):
    x1,x2=x
    f1 = np.array([np.log((x1)**2+(x2)**2)-np.sin((x1)*(x2))-np.log(2*np.pi),                    np.exp((x1)-(x2)) + np.cos((x1)*(x2))])
    return f1
def J1(x):
    x1,x2=x
    return np.array([[2*x1/(x1**2 + x2**2) - x2*np.cos(x1*x2),-x1*np.cos(x1*x2) + 2*x2/(x1**2 + x2**2)],                    [-x2*np.sin(x1*x2) + np.exp(x1 - x2),-x1*np.sin(x1*x2) - np.exp(x1 - x2)]])
x0=[aprox_inicial_f1[0][0],aprox_inicial_f1[1][0]]
raiz,iteraciones=NR_mod_multi(F1,J1,x0,50,1e-6)


# In[7]:


print("newton modificado")
print("aprox_fina_f1: ", raiz)
print("iteraciones:",iteraciones)


# In[8]:


#sistema de ecuaciones 2
# para hallar una x0,y0 inicial
#gradiente descendente
f3=["x**3 + y*(x**2) - x*z"]
f4=["exp(x) + exp(y) - z"]
f5=["y**2 - 2*x*z -4"]
f2_cuad = ['(x**3 + y*(x**2) - x*z)**2+(exp(x) + exp(y) - z)**2+(y**2 - 2*x*z -4)**2']
variables_2 = 'x y z'
valor_inicial_2 = np.array([[0], [0], [0]]) #aprox fila 9 de df
aprox_inicial_f2,iter2_gd = FG.sec_gradiente(valor_inicial_2, f2_cuad, variables_2, 50,0.05)
print("sistema de ecuaciones 2:")
print("gradiente descendente:")
print("aprox_inicial_f2: ", aprox_inicial_f2)
print("iteraciones:",iter2_gd)


# In[9]:


#newton modificado
def F2(x):
    x,y,z = x
    f1 = np.array([x**3 + y*(x**2) - x*z, np.exp(x) +np.exp(y) -z, y**2 - 2*x*z -4 ])
    return f1
def J2(x):
    x,y,z = x
    return np.array([[3*(x**2) + 2*x*y - z, x**2, -x], [np.exp(x), np.exp(y),-1], [-2*z, 2*y, -2*x]])
x0 = [aprox_inicial_f2[0][0], aprox_inicial_f2[1][0], aprox_inicial_f2[2][0]]
raiz_2,iteraciones_2 = NR_mod_multi(F2,J2,x0,50,1e-6)


# In[10]:


print("newton modificado")
print("aprox_fina_f2: ", raiz_2)
print("iteraciones:",iteraciones_2)


# In[11]:


actividad_1 = pd.DataFrame()
actividad_1["sistema"] = [1,1,2,2,2]
actividad_1["variable"] =["x","y","x","y","z"]
actividad_1["gradiente descendente"] = [aprox_inicial_f1[0][0],aprox_inicial_f1[1][0],                                       aprox_inicial_f2[0][0],aprox_inicial_f2[1][0],aprox_inicial_f2[2][0]]
actividad_1["newton modificado"] = [raiz[0],raiz[1],raiz_2[0],raiz_2[1],raiz_2[2]]
actividad_1["iter NM"] = [iteraciones,iteraciones,iteraciones_2,iteraciones_2,iteraciones_2]
actividad_1["iter GD"] = [iter1_gd,iter1_gd,iter2_gd,iter2_gd,iter2_gd]


# In[13]:


print(actividad_1)


# In[ ]:




