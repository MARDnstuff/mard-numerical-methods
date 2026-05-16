import numpy as np
from numpy import sin, cos, exp, pi
from .baseFunction import Function
import logging


logger = logging.getLogger(__name__)

class EasomFunction(Function):
    def __init__(self) -> None:
        """
        Constructor
        """
        domain = (-2*pi, 2*pi)
        self.global_min = (pi, pi)
        self.tol = 1e-3
        super().__init__(domain)

    def f(self, x: np.array) -> float:
        """
        Función objetivo a minimizar

        :param x: vector de variables [x0, x1]
        :return: Valor escalar de la función evaluada en x.
        """
        if len(x) != 2:
            raise ValueError("Easom solo admite 2 dimensiones")
        self.f_count_invok += 1
        return -1*cos(x[0])*cos(x[1])*exp(-1*(x[0] - pi)**2 - (x[1] - pi)**2)

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
            raise ValueError("Easom solo admite 2 dimensiones")
        
        tx1 = cos(x[1])*exp(-1*(x[0] - pi)**2 - (x[1] - pi)**2)*(sin(x[0]) + 2*(x[0] - pi)*cos(x[0]))
        tx2 = cos(x[0])*exp(-1*(x[0] - pi)**2 - (x[1] - pi)**2)*(sin(x[1]) + 2*(x[1] - pi)*cos(x[1]))
        self.Df_count_invok += 1
        return np.array([tx1, tx2])
    

    def H(self, x: np.array) -> np.array:
        """
        Matriz Hessiana

        NOTA:
        - 2 variables
        
        :param x: Punto en el que se evalúa el gradiante [x0, x1, ...]
        :return: Matrix evaluada en x.
        """

        if len(x) != 2:
            raise ValueError("Easom solo admite 2 dimensiones")

        x1 = x[0]
        x2 = x[1]

        E = exp(
            -((x1 - pi)**2 + (x2 - pi)**2)
        )

        # Segunda derivada respecto a x
        h11 = (
            cos(x2) * E * (
                4 * (x1 - pi) * sin(x1)
                + (2 - 4 * (x1 - pi)**2) * cos(x1)
            )
        )

        # Segunda derivada respecto a y
        h22 = (
            cos(x1) * E * (
                4 * (x2 - pi) * sin(x2)
                + (2 - 4 * (x2 - pi)**2) * cos(x2)
            )
        )

        # Derivada cruzada
        h12 = (
            E * (
                sin(x1) * sin(x2)
                + 2 * (x2 - pi) * sin(x1) * cos(x2)
                + 2 * (x1 - pi) * cos(x1) * sin(x2)
                + 4 * (x1 - pi) * (x2 - pi) * cos(x1) * cos(x2)
            )
        )

        self.H_count_invok += 1
        return np.array([
            [h11, h12],
            [h12, h22]
        ])