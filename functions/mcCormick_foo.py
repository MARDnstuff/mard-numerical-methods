import numpy as np
from numpy import sin, cos
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class McCormickFunction(Function):
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-1.5, 4)
        super().__init__(domain)
        self.domain_x2 = (-3, 4)

    def f(self, x: np.array) -> float:
        """
        Función objetivo a minimizar

        :param x: vector de variables [x0, x1]
        :return: Valor escalar de la función evaluada en x.
        """
        if len(x) != 2:
            raise ValueError("McCormick solo admite 2 dimensiones")

        return sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

    def Df(self, x: np.array) -> np.array:
        """
        Gradiente (vector de derivadas parciales) de la función objetivo f
        
        Nota:
        - El gradiente apunta en la dirección de mayor crecimiento de f.
        - El descenso del gradiente se mueve en dirección opuesta para minimizar.

        :param x: Punto en el que se evalúa el gradiante [x0, x1]
        :return: Vector evaluado en x.
        """
        if len(x) != 2:
            raise ValueError("McCormick solo admite 2 dimensiones")
        res = [cos(x[0] + x[1]) + 2*(x[0] - x[1]) - 1.5, cos(x[0] + x[1]) + 2*(x[0] - x[1]) + 2.5]
        return np.array(res)
    
    def H(self, x: np.array) -> np.array:
        """
        Matriz Hessiana
        
        :param x: Punto en el que se evalúa el gradiante [x0, x1, ...]
        :return: Matrix evaluada en x.
        """
        if len(x) != 2:
            raise ValueError("McCormick solo admite 2 dimensiones")
        s = -sin(x[0] + x[1])
        
        h = np.array([
            [s + 2, s - 2],
            [s - 2, s + 2]
        ]) 
        return h