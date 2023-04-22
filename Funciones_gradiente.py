# Métodos Numéricos
# Clase Práctica 5
# Implementación de la interpolación de Lagrange para 3 puntos
# Clase Práctica 6
# El método del gradiente descendiente

import copy as cp
import numpy as np
import sympy as sym
import numpy.polynomial.polynomial as polyfunc
from sympy.vector import gradient


###FUNCIONES IGUALES EN EL MÉTODO DE NEWTON_RAPHSON
def gauss_piv(a, b):
    ab = np.concatenate((a, b), axis=1)  # creamos la matrix extendida
    tam = np.size(b)  # con la cantidad de elementos de b sabemos la cantidad de filas y columnas
    for k in range(tam):
        ab_col = ab[k:tam, k]  # agarramos la columna desde el índice hasta el final
        m_col = np.argmax(np.abs(ab_col))  # vemos el argumento del mayor valor absoluto
        if m_col > k:
            fil_aux1 = cp.copy(ab[k, :])
            # debemos hacer "copy" porque si ponemos fil_aux1 = ab[k, :] enlaza sus valores
            ab[k, :] = ab[m_col, :]  # ponemos la fila del argumento con mayor valor absoluto arriba
            ab[m_col, :] = fil_aux1
        for i in range(k + 1, tam, 1):  # en range() tenemos (inicio, fin, paso)
            m = ab[i, k] / ab[k, k]
            ab[i, k] = 0
            for j in range(k + 1, tam + 1, 1):
                ab[i, j] = ab[i, j] - m * ab[k, j]
        # Aquí terminamos el pivotamiento y vamos a la sustitución
    x = cp.copy(b)  # creamos un vector x de la misma dimensión de b
    x[:, 0] = 0  # lo ceramos
    x[tam - 1, 0] = ab[tam - 1, tam] / ab[tam - 1, tam - 1]
    for k in range(tam - 2, -1, -1):
        s = 0.
        for j in range(k + 1, tam, 1):
            s = s + ab[k, j] * x[j, 0]
            x[k, 0] = (ab[k, tam] - s) / ab[k, k]
    return x


def jacobiano(v_str, f_list):
    '''Recibimos un string con las variables
    y una lista con las ecuaciones en forma de
    cadenas de caracteres'''
    # Extraido de https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
    vars = sym.symbols(v_str)
    f = sym.sympify(f_list)
    J = sym.zeros(len(f), len(vars))
    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i, j] = sym.diff(fi, s)
    return J


def evaluar_matriz(matriz, str1, vect_eval):
    '''
    :param matriz: matriz o vector de sympy
    :param str1: string con las variables
    :param vect_eval: vector columna con los valores a evaluar
    :return:la matriz evaluada de sympy
    '''
    var_list = sym.symbols(str1)  # creamos una lista con las variables
    len_string = len(var_list)
    i = 0
    mat_aux = cp.copy(matriz)
    while i < len_string:
        mat_aux = mat_aux.subs(var_list[i], vect_eval[i, 0])
        i = i + 1
    vect_col = np.zeros([len_string, 1])
    for i in range(len_string):
        vect_col[i, 0] = mat_aux[i]
    return vect_col


def evaluar_funcion(F, str1, vect_eval):
    '''
    :param F: Función como lista
    :param str1: string con las variables
    :param vect_eval: vector columna con los valores a evaluar
    :return: La función evaluada en un punto (flotante)
    '''
    var_list = sym.symbols(str1)
    f = sym.sympify(F)
    tam = len(var_list)
    for i in range(tam):
        f[0] = f[0].subs(var_list[i], vect_eval[i, 0])
    fx = f[0]
    return fx


def stringaVector(x):
    '''
    :param x: vector en forma de string, componentes separadas por espacios
    :return: vector columna de numpy
    '''
    x_list = x.split()
    longitud = len(x_list)
    vector = np.zeros([longitud, 1])
    for i in range(longitud):
        vector[i, 0] = x_list[i]
    return vector


###FUNCIONES IGUALES EN EL MÉTODO DE NEWTON_RAPHSON

def lag_polinomio3(x, y):
    '''
    :param x: valores en x de los puntos
    :param y: valores en y de los puntos
    :return: sumapolins (polinomio interpolador), derivada (su derivada), soldig0 (la solución de la derivada igualada a 0)
    '''
    N = np.size(x)
    div = np.zeros(N)
    mult = np.zeros(N)
    matriz_binms = np.ones([N, 2])
    matriz_binms[:, 0] = -1 * x[:, 0]
    sumapolins = [0]  # Creamos el polinomio vacío para ir sumando
    for i in range(N):
        '''Formamos los valores g(xi)/Productoria(xi-xj)
        y también los productos Productoria(x-xj)'''
        div[i] = 1
        producto_i = [1]
        for j in range(N):
            if i != j:
                div[i] = div[i] * (x[i] - x[j])
                xj = matriz_binms[j, :]
                producto_i = polyfunc.polymul(producto_i, xj)
        mult[i] = y[i] / div[i]
        producto_i = mult[i] * producto_i
        sumapolins = sumapolins + producto_i
    '''Recordamos que para los polinomios se leerá de derecha a izquierda
    como el coeficiente de los exponentes de x en forma ascendente,
    y estos son por lo general vectores fila de numpy
    Por ejemplo [3 2 -1] es -x**2+2*x-1 y [2 -2] es -2*x+2'''
    ''' A partir de aquí los cálculos serán solo para
    El método del gradiente, no para interpolación en general'''
    derivada = np.zeros(N - 1)
    for i in range(N - 1):
        derivada[i] = (i + 1) * sumapolins[i + 1]
    segunda_derivada = np.zeros(N - 2)
    for i in range(N - 2):
        segunda_derivada[i] = (i + 1) * derivada[i + 1]
    if segunda_derivada[0] != 0:
        soldig0 = (-1 * derivada[0]) / derivada[1]
    else:
        soldig0 = x[2, 0]
    return sumapolins, derivada, soldig0


def t_paso_desc(variables, F, x):
    '''
    :param variables: variables de las funciones como string separadas por espacios
    :param F: función en sympy, lista de un elemento
    :param x: punto que no es un minimizador, vector columna, se puede usar (0, 0, 0)
    :return:
    '''
    tam = np.size(x)
    x0 = cp.copy(x)
    t = np.array([[0.], [0.], [0.]]) #Cuidado al definir t, deben ser flotantes
    t[0, 0] = 0.
    nablaF = jacobiano(variables, F)
    nablaF_evaluado = evaluar_matriz(nablaF, variables, x)
    # De la función se obtiene un vector nablaF de 3 componentes
    x_gt1 = x0 - (t[0] * nablaF_evaluado)
    gt1 = evaluar_funcion(F, variables, x_gt1)
    cont = 1
    t[2, 0] = t[0, 0] + 0.001
    x_gt3 = x0 - (t[2, 0] * nablaF_evaluado)
    gt3 = evaluar_funcion(F, variables, x_gt3)
    while gt3 >= gt1:
        t_pruebamas = t[0, 0] + (cont/1000.)
        t_pruebamenos = t[0, 0] - (cont/1000.)
        x_gtmas = x0 - (t_pruebamas * nablaF_evaluado)
        x_gtmenos = x0 - (t_pruebamenos * nablaF_evaluado)
        gtmas = evaluar_funcion(F, variables, x_gtmas)
        gtmenos = evaluar_funcion(F, variables, x_gtmenos)
        if gtmas < gt1:
            t[2, 0] = t_pruebamas
            x_gt3 = x_gtmas
            gt3 = gtmas
        elif gtmenos < gt1:
            t[2, 0] = t_pruebamenos
            x_gt3 = x_gtmenos
            gt3 = gtmenos
        else:
            cont = cont + 1
    t[1, 0] = t[2, 0] / 2
    x_gt2 = x0 - (t[1, 0] * nablaF_evaluado)
    gt2 = evaluar_funcion(F, variables, x_gt2)
    gt = np.array([[gt1], [gt2], [gt3]])
    sumapolins, derivada, tk = lag_polinomio3(t, gt)
    return tk


def sec_gradiente(x0, F, vars, N_iter, tol):
    '''
    :param x0: Punto inicial (no minimizador)
    :param F: Función (lista, con una cadena de caracteres como único elemento)
    :param vars: Variables (cadena de caracteres, variables separadas por espacios)
    :param N_iter: Número de iteraciones (entero)
    :return: El punto más cercano al mínimo local obtenido con el método del gradiente
    '''
    x = cp.copy(x0)  # aquí guardaremos nuestras aproximaciones
    iter = 1
    cumple = False
    while (not cumple and iter <= N_iter):
        tk = t_paso_desc(vars, F, x)
        nablaF = jacobiano(vars, F)
        nablaF_evaluado = evaluar_matriz(nablaF, vars, x)
        x = x - tk * nablaF_evaluado
        iter = iter + 1
        cumple = abs(evaluar_funcion(F, vars, x)) <= tol
    if iter < N_iter:
        return x, iter
    else:
        return 'El sistema no converge'
        
