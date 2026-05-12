import numpy as np
from numpy import sin, cos, exp, pi
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class RastriginFunction(Function):
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
        d = len(x)
        res = 10*d
        for i in range(0, d):
            res += 2*(x[i]**2) - 10*cos(2*pi*x[i])
        self.f_count_invok += 1
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
        
        res = []
        for xi in x:
            term = 2*xi + 20*pi*sin(2*pi*xi) 
            res.append(term)
        self.Df_count_invok += 1
        return np.array(res)

    def H(self, x: np.array) -> np.array:
        """
        Matriz Hessiana
        
        :param x: Punto en el que se evalúa el gradiante [x0, x1, ...]
        :return: Matrix evaluada en x.
        """
        diag = (
            2
            + 40 * (np.pi**2) * np.cos(2 * np.pi * x)
        )
        self.H_count_invok += 1
        return np.diag(diag)