# Métodos Numéricos
# Clase Práctica 2
# Implementación de transformaciones de Householder
# Para factorización QR
import numpy as np
import copy as cp


# Tenemos una matriz A y queremos descomponerla en A=QR
# Lo primero es hacer la función "Transformaciones de Householder"
def trans_householder(a):
    # Formamos el reflector
    (tamx, tamy) = np.shape(a)
    ident = np.eye(tamx)
    ax = np.zeros((tamx, 1)) # Construimos un vector columna de dimensión conocida de esta manera
    for i in range(tamx):
        ax[i, 0] = a[i, 0]
    e = cp.copy(ax)
    e[:] = 0
    e[0] = 1 # Construimos el vector e de dimensión "tamx"
    u = ax - (np.linalg.norm(ax) * e)
    r = ident - (2 * (u @ np.transpose(u)) / (np.transpose(u) @ u))
    return(r)
# Luego hacemos una función para la factorización QR con transformaciones de Householder
def QR_householder(a):
    (tamx, tamy) = np.shape(a)
    a_trans = cp.copy(a)
    mat_mult_R = np.eye(tamx)
    for i in range(tamx-1): #vamos hasta n-1 en una matriz nxn
        ri = trans_householder(a_trans[i:tamx, i:tamx])
        (tamRix, tamRiy) = np.shape(ri)
        if tamRix < tamx:
            Riaux = np.zeros((tamx, tamy))
            Riaux[i:tamx, i:tamx] = ri
            for dp in range(tamx - tamRix):
                Riaux[dp, dp] = 1
            ri = Riaux
        a_trans = ri @ a_trans
        mat_mult_R = ri @ mat_mult_R # Al final el resultado es la matriz "P"
    # Una vez terminamos el for pasamos a definir Q y R
    q = np.transpose(mat_mult_R)
    r = mat_mult_R @ a
    p = cp.copy(mat_mult_R)
    t = cp.copy(r)
    return(q, r, p, t)
