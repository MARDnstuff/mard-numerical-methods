import numpy as np


def MS(df: callable, x0: float, x1: float, tol: float = 1e-4, max_iter: int = 20) -> float:
    """
    Método de la Secante para encontrar raíces de una función.

    :param df:       Función cuya raíz se busca (derivada de la función de línea).
    :param x0:       Primer punto inicial.
    :param x1:       Segundo punto inicial.
    :param tol:      Tolerancia para el criterio de convergencia. Por defecto 1e-4.
    :param max_iter: Número máximo de iteraciones. Por defecto 20.

    :return: Aproximación a la raíz de df.
    """
    xk  = x0
    xk1 = x1
    for _ in range(max_iter):
        xa  = xk1
        xk1 = (df(xk1)*xk - df(xk)*xk1) / (df(xk1) - df(xk))
        if abs(xk1 - xa) < tol:
            break
        xk = xa
    return xk1


def df_adelante(f: callable, x: float, h: float) -> float:
    """
    Aproximación de la derivada por diferencias finitas hacia adelante.

    Calcula:
        f'(x) ≈ ( f(x+h) - f(x) ) / h

    :param f: Función a derivar.
    :param x: Punto en el que se evalúa la derivada.
    :param h: Tamaño del paso. Valores típicos: 1e-4 a 1e-6.

    :return: Aproximación numérica de f'(x).
    """
    return (f(x + h) - f(x)) / h


def MDG_DM(f: callable, Df: callable, x0: np.array, max_iter: int = 20) -> tuple[np.array, np.array]:
    """
    Método de Descenso del Gradiente con Búsqueda de Línea Exacta (Máximo Descenso).

    En cada iteración busca el paso óptimo alpha que minimiza f en la
    dirección del gradiente, resolviendo:
        min_{alpha} g(alpha) = f(x - alpha * Df(x))

    La derivada de g se aproxima con diferencias finitas y su raíz
    se encuentra con el Método de la Secante (MS).

    La regla de actualización es:
        x_{k+1} = x_k - alpha_opt * Df(x_k)

    :param f:        Función objetivo.
    :param Df:       Función que calcula el gradiente de f.
    :param x0:       Punto inicial desde donde comienza la optimización.
    :param max_iter: Número máximo de iteraciones. Por defecto 20.

    :return: Tupla (x, X) donde:
        - x: np.array con la aproximación al mínimo encontrado.
        - X: np.array shape (max_iter+1, n) con la trayectoria completa de puntos visitados, incluyendo x0.
    """
    x = x0
    X = [x0.copy()]

    for _ in range(max_iter):
        g  = lambda alph: f(x - alph * Df(x0))
        dg = lambda alph: df_adelante(g, alph, 0.0001)

        alp_opt = MS(dg, 0, 1) # alpha óptimo via método de la secante
        x -= alp_opt * Df(x)
        X.append(x.copy())

    return x, np.array(X)