import numpy as np
from .baseFunction import Function


class SphereFunction(Function):
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-5.12, 5.12)
        super().__init__(domain)

    def f(self, x: np.array) -> float:
        """
        Función objetivo a minimizar

        :param x: vector de variables [x0, x1]
        :return: Valor escalar de la función evaluada en x.
        """
        return np.sum(x**2, axis=0)

    def Df(self, x: np.array) -> np.array:
        """
        Gradiente (vector de derivadas parciales) de la función objetivo f
        
        Nota:
        - El gradiente apunta en la dirección de mayor crecimiento de f.
        - El descenso del gradiente se mueve en dirección opuesta para minimizar.

        :param x: Punto en el que se evalúa el gradiante [x0, x1]
        :return: Vector evaluado en x.
        """
        res = []
        for elem in x:
            # sumatoria de 2xi, donde i varia deste 0 hasta el tamaño de x
            res.append(2*elem)
        return np.array(res)

    def H(self, x: np.array) -> np.array:
        """
        Matriz Hessiana
        
        :param x: Punto en el que se evalúa el gradiante [x0, x1, ...]
        :return: Matrix evaluada en x.
        """
        return 2*np.eye(len(x))