import numpy as np
import matplotlib.pyplot as plt

# TODO: Agregar a cada función su dominio correspondiente
# TODO: Agregar el conteo del llamado de la función 
# TODO: Generar una matriz para comparar
def _busqueda_wolfe(f: callable, Df: callable, x: np.array, d: np.array, c1: float = 1e-4, c2: float = 0.9, alph_max: float = 1.0, max_iter: int = 100) -> float:
    """
    Búsqueda de línea que satisface las condiciones fuertes de Wolfe.

    Implementa el algoritmo de bisección/expansión para encontrar un paso
    alpha que cumpla Armijo (suficiente descenso) y curvatura fuerte.

    :param f:        Función objetivo.
    :param Df:       Gradiente de f.
    :param x:        Punto actual.
    :param d:        Dirección de descenso (tipicamente -Df(x)).
    :param c1:       Constante Armijo (0 < c1 < c2 < 1). Por defecto 1e-4.
    :param c2:       Constante curvatura fuerte (c1 < c2 < 1). Por defecto 0.9.
    :param alph_max: Paso máximo permitido. Por defecto 1.0.
    :param max_iter: Máximo de iteraciones internas. Por defecto 100.
    :return:         Paso alpha que satisface las condiciones de Wolfe.
    """
    alph_prev = 0.0
    alph_curr = alph_max
    f0    = f(x)
    phi0  = np.dot(Df(x), d)   # pendiente inicial: ∇f(x) d

    for i in range(max_iter):
        f_curr = f(x + alph_curr * d)

        # Viola Armijo o aumentó respecto al paso anterior -> acotar y zoom
        if f_curr > f0 + c1 * alph_curr * phi0 or (i > 0 and f_curr >= f(x + alph_prev * d)):
            return _zoom(f, Df, x, d, alph_prev, alph_curr, f0, phi0, c1, c2)

        phi_curr = np.dot(Df(x + alph_curr * d), d)

        # Satisface curvatura fuerte -> aceptar
        if abs(phi_curr) <= c2 * abs(phi0):
            return alph_curr

        # Pendiente positiva -> acotar y zoom
        if phi_curr >= 0:
            return _zoom(f, Df, x, d, alph_curr, alph_prev, f0, phi0, c1, c2)

        alph_prev = alph_curr
        alph_curr = min(2 * alph_curr, alph_max)

    return alph_curr


def _zoom(f: callable, Df: callable, x: np.array, d: np.array,alph_lo: float, alph_hi: float, f0: float, phi0: float, c1: float, c2: float, max_iter: int = 50) -> float:
    """
    Fase de zoom del algoritmo de Wolfe.

    Reduce el intervalo [alph_lo, alph_hi] hasta encontrar un alpha
    que satisfaga ambas condiciones fuertes de Wolfe.

    :param f:       Función objetivo.
    :param Df:      Gradiente de f.
    :param x:       Punto actual.
    :param d:       Dirección de descenso.
    :param alph_lo: Extremo inferior del intervalo.
    :param alph_hi: Extremo superior del intervalo.
    :param f0:      Valor f(x) en el punto actual.
    :param phi0:    Pendiente inicial ∇f(x)ᵀ d.
    :param c1:      Constante Armijo.
    :param c2:      Constante curvatura fuerte.
    :param max_iter: Máximo de iteraciones. Por defecto 50.
    :return:        Paso alpha que satisface las condiciones de Wolfe.
    """
    for _ in range(max_iter):
        alph = (alph_lo + alph_hi) / 2.0    # bisección
        f_alph = f(x + alph * d)

        if f_alph > f0 + c1 * alph * phi0 or f_alph >= f(x + alph_lo * d):
            alph_hi = alph
        else:
            phi_alph = np.dot(Df(x + alph * d), d)

            if abs(phi_alph) <= c2 * abs(phi0):
                return alph                 

            if phi_alph * (alph_hi - alph_lo) >= 0:
                alph_hi = alph_lo

            alph_lo = alph

    return alph


def MDG_Wolfe(f: callable, Df: callable, x0: np.array, c1: float = 1e-4, c2: float = 0.9, alph_max: float = 1.0, max_iter: int = 20) -> tuple[np.array, np.array]:
    """
    Método de Descenso del Gradiente con búsqueda de línea (condiciones fuertes de Wolfe).

    En cada iteración busca un paso alpha que satisfaga:
        - Armijo:            f(x - alpha*∇f) ≤ f(x) - c1*alpha*||∇f||²
        - Curvatura fuerte:  |∇f(x - alpha*∇f)ᵀ ∇f(x)| ≤ c2*||∇f||²

    A diferencia de MDG_PF, el paso alpha varía en cada iteración.

    :param f:        Función objetivo.
    :param Df:       Función que calcula el gradiente.
    :param x0:       Punto inicial desde donde comienza la optimización.
    :param c1:       Constante Armijo (0 < c1 < c2 < 1). Por defecto 1e-4.
    :param c2:       Constante curvatura fuerte (c1 < c2 < 1). Por defecto 0.9.
    :param alph_max: Paso máximo permitido en la búsqueda. Por defecto 1.0.
    :param max_iter: Número máximo de iteraciones. Por defecto 20.

    :return: Tupla (x, X) donde:
            - x: arreglo con la aproximación al mínimo encontrado.
            - X: arreglo shape (max_iter+1, n) con la trayectoria completa.
    """
    x = x0.copy()
    X = [x0.copy()]

    for _ in range(max_iter):
        g  = Df(x)
        # dirección de descenso
        d  = -g 

        # Buscar alpha que satisfaga Wolfe fuerte
        alph = _busqueda_wolfe(f, Df, x, d, c1, c2, alph_max)

        # actualización
        x = x + alph * d
        X.append(x.copy())

    return x, np.array(X)


