from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):
    """
    Clase abstracta base para clases tipo Función

    Link: https://www.sfu.ca/~ssurjano/optimization.html
    """

    def __init__(self, domain: tuple[float, float]) -> None:
        """
        Constructor
        
        """
        self.domain = domain

    @abstractmethod
    def f(self, x: np.array) -> float:
        """
        Función objetivo a minimizar

        :param x: vector de variables [x0, x1]
        :return: Valor escalar de la función evaluada en x.
        """
        pass
    
    @abstractmethod
    def Df(self, x: np.array) -> np.array:
        """
        Gradiente (vector de derivadas parciales) de la función objetivo f
        
        Nota:
        - El gradiente apunta en la dirección de mayor crecimiento de f.
        - El descenso del gradiente se mueve en dirección opuesta para minimizar.

        :param x: Punto en el que se evalúa el gradiante [x0, x1]
        :return: Vector evaluado en x.
        """
        pass
    
    @abstractmethod
    def H(self, x: np.array) -> np.array:
        """
        Matriz Hessiana
        
        :param x: Punto en el que se evalúa el gradiante [x0, x1, ...]
        :return: Matriz evaluada en x.
        """
        pass
