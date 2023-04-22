#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# metodo Newton modificado
def NR_mod_multi(F, J, x, imax, tol):
    cumple = False
    k = 0
    J0 = J(x)
    while(not cumple and k<imax):
        deltax = np.linalg.solve(J0,-F(x))
        x = x + deltax
        print(f'iteracion:{k}-->{x}')
        cumple = np.linalg.norm(F(x)) <= tol
        k += 1
    if k < imax:
        return x,k
    else:
        return 'El sistema no converge',k


# In[ ]:




