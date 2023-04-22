#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# In[61]:


PHI = np.pi/4
A = 5.0
M = 24.6
C = 0.5
K = 400.0
W1 = 4.0


# In[62]:


# resolviendo la ecuacion implicita
def W(w,h=PHI,c=C,m=M,k=K):
    return m*(w**2) + (c/np.tan(h))*w - k

def F(f,A=A,w=W1,c=C,m=M,k=K):
    return f**2 - (A**2)*((k - m*(w**2))**2 + (c**2)*(w**2))


# In[73]:


w = fsolve(W,100)
f = fsolve(F,1)


# In[87]:


print("valores iniciales: ")
print("A:",A,",PHI:",PHI,",m:",M,",c:",C,",k:",K,",w1:",W1)
print("resultado:")
print("w=",w,"F=",f)


# In[66]:


# introduciendo los valores
print("ingrese phi:")
PHI = float(input())
print("ingrese A:")
A = float(input())
print("ingrese m:")
M = float(input())
print("ingrese c:")
C = float(input())
print("ingrese k:")
K = float(input())
print("ingrese w1:")
W1 = float(input())


# In[81]:


def ff(A,w,c,m,k):
    return A*np.sqrt((k-m*(w)**2)**2 + (c**2)*(w**2))
def ww(h,c,m,k):
    w1=((c/np.tan(h)) + np.sqrt((c/np.tan(h))**2 + 4*m*k))/(2*m)
    w2=((c/np.tan(h)) - np.sqrt((c/np.tan(h))**2 + 4*m*k))/(2*m)
    return w1,w2


# In[84]:


f_2 = ff(A,W1,C,M,K)
w_2 = ww(PHI,C,M,K)


# In[85]:


print("w=",w_2,"F=",f_2)


# In[ ]:




