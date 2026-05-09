import numpy as np

def NewtonMethod(G: callable, H: callable, x0: np.array, tol:float = 10**(-4), max_iter: int =20) -> np.array:
    """
    Método de Newton para optimización multivariable.

    Este método busca aproximar un punto crítico de una función objetivo
    utilizando información de primer y segundo orden:
    
    - Gradiente: G(x)
    - Hessiana: H(x)

    En cada iteración se actualiza el punto mediante:

        x_{k+1} = x_k - H(x_k)^(-1) G(x_k)

    donde:
    - G(x_k) es el gradiente evaluado en x_k
    - H(x_k) es la matriz Hessiana evaluada en x_k

    El algoritmo termina cuando:
    - La norma de la diferencia entre iteraciones consecutivas es menor que la tolerancia `tol`, o
    - Se alcanza el número máximo de iteraciones.

    :param G: Función objetivo gradiente
    :param H: Función de la Matriz Hessiana 
    :param x0: Punto inicial del algoritmo.
    :param tol: Tolerancia para el criterio de convergencia. Valor por defecto: 1e-4.
    :param max_iter: Número máximo de iteraciones permitidas. Valor por defecto: 20.

    Notes
    -----
    - El método de Newton posee convergencia cuadrática cerca de la solución si la Hessiana es no singular.
    - Requiere calcular e invertir la Hessiana en cada iteración.
    - Puede divergir si el punto inicial está lejos del óptimo o si la Hessiana no es definida positiva.

    """

    x = x0
    X = [x0.copy()]

    for _ in range(max_iter):
        xa = x

        x = x - np.dot(
            np.linalg.inv(H(x)),
            G(x)
        )
        X.append(x.copy())

        if np.linalg.norm(x - xa) < tol:
            break

    return x, np.array(X)