import numpy as np
from numpy import sin, cos
from .baseFunction import Function
import logging

logger = logging.getLogger(__name__)

class DixonPriceFunction(Function):
    def __init__(self) -> None:
        """
        Constructor
        """
        pass

    def f(self, x: np.array) -> float:
        """
        Función objetivo a minimizar

        :param x: vector de variables [x0, x1]
        :return: Valor escalar de la función evaluada en x.
        """

        res = (x[0] - 1)**2

        for i in range(2, len(x) + 1):
            res += i*(2*(x[i - 1])**2 - x[i - 2])**2
        return res

    def Df(self, x: np.array) -> np.array:
        """
        Gradiente (vector de derivadas parciales) de la función objetivo f
        
        Nota:
        - El gradiente apunta en la dirección de mayor crecimiento de f.
        - El descenso del gradiente se mueve en dirección opuesta para minimizar.

        :param x: Punto en el que se evalúa el gradiante [x0, x1]
        :return: Vector evaluado en x.
        """
        n = len(x)

        if n == 1:
            return np.array([2 * (x[0] - 1)])

        res = [6*x[0] - 8*(x[1]**2) - 2]
        
        for i in range(2, n):
            term = 8*(i -1)*x[i - 1]*(2*(x[i - 1]**2) - x[i - 2]) - 2*i*(2*(x[i]**2) - x[i - 1])
            res.append(term)
        
        res.append(8*n*x[n - 1]*(2*(x[n - 1]**2) - x[n - 2]))

        return np.array(res)
    
    def H(x: np.array) -> np.array:
        """
        Matriz Hessiana
        
        :param x: Punto en el que se evalúa el gradiante [x0, x1, ...]
        :return: Matrix evaluada en x.
        """

        n = len(x)

        H = np.zeros((n, n))

        # H[0,0]
        H[0, 0] = 6

        if n > 1:
            H[0, 1] = -16 * x[1]
            H[1, 0] = -16 * x[1]

        # Parte intermedia
        for i in range(1, n - 1):

            # Diagonal principal
            H[i, i] = (
                48 * (i + 1) * (x[i] ** 2)
                - 8 * (i + 1) * x[i - 1]
                + 2 * (i + 2)
            )

            # Derivadas cruzadas
            cross = -8 * (i + 1) * x[i]

            H[i, i - 1] = cross
            H[i - 1, i] = cross

        # Última diagonal
        if n > 1:
            H[n - 1, n - 1] = (
                48 * n * (x[n - 1] ** 2)
                - 8 * n * x[n - 2]
            )

            cross = -8 * n * x[n - 1]

            H[n - 1, n - 2] = cross
            H[n - 2, n - 1] = cross

        return H