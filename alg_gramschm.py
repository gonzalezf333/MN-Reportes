# Métodos Numéricos
# Clase Práctica 2
# Implementación del algoritmo de Gram-Schmidt
# Para factorización QR
import numpy as np
import copy as cp
# Tenemos una matriz A y queremos descomponerla en A=QR
def QR_gramschmidt(a):
    # Vemos como vectores a las columnas de la matriz A
    q = cp.copy(a)
    q[:, :] = 0 # hacemos a q una matriz de ceros
    r = cp.copy(q) # hacemos a r una matriz de ceros
    (tamx, tamy) = np.shape(a)
    # Formamos la matriz Q
    u0 = cp.copy(a[:, 0]) # Agarramos la primera columna
    r[0, 0] = np.linalg.norm(a[:, 0]) # La primera componente de r
    u0 = u0 / r[0, 0]
    q[:, 0] = u0 # La primera columna de q
    for k in range(1, tamx):
        vk = cp.copy(a[:, k])
        sumatoria = 0
        for j in range(0, k, 1):
            r[j, k] = q[:, j] @ vk # siempre multiplica vk
            # vamos formando a la par la matriz r
            prod = r[j, k] * q[:, j]
            sumatoria = sumatoria + prod
        uk = vk - sumatoria
        r[k, k] = np.linalg.norm(uk)# formamos los componentes
        # dentro de la diagonal principal de r
        uk = uk / r[k, k]
        q[:, k] = uk
        p = np.transpose(q)
        t = cp.copy(r)
    return(q, r, p, t)