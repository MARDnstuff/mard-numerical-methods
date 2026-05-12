import numpy as np
import math
from config.logging import setUpLogging
from graph.graph_foo import FunctionPlotter2D
from numerical_methods.gradient_descent import MDG_Wolfe
from numerical_methods.newton import NewtonMethod
from numerical_methods.max_decrease import MDG_DM
from functions.sphere_foo import SphereFunction
from functions.mcCormick_foo import McCormickFunction
from functions.dixon_price_foo import DixonPriceFunction
from functions.easom_foo import EasomFunction
from functions.rastrigin_foo import RastriginFunction
import logging

# Logging
setUpLogging()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    # Sphere
    # Dominio típico: [-5.12, 5.12]
    sphere = SphereFunction()
    x0_sphere = np.array([2.84, -4.11])

    # McCormick
    # Dominio típico:
    # x ∈ [-1.5, 4]
    # y ∈ [-3, 4]
    mcCormick = McCormickFunction()
    x0_mccormick = np.array([1.27, -1.92])

    # Easom
    # Dominio típico: [-100, 100]
    easom = EasomFunction()
    x0_easom = np.array([12.45, 8.73])
    print(np.array([[1,2], [2, 2]]))

    # Dixon-Price
    # Dominio típico: [-10, 10]
    dixon_price = DixonPriceFunction()
    x0_dixon_price = np.array([-3.58, 6.21])

    # Rastrigin
    # Dominio típico: [-5.12, 5.12]
    rastrigin = RastriginFunction()
    x0_rastrigin = np.array([0.1, 0.1])

    
    # x_opt, trayectoria = MDG_Wolfe(
    #     f        = easom.f,
    #     Df       = easom.Df,
    #     x0       = x0_easom,
    #     c1       = 1e-4,    # qué tan estricto es el descenso (Armijo)
    #     c2       = 0.9,     # qué tan estricta es la curvatura
    #     alph_max = 1.0,     # paso máximo que puede probar la búsqueda
    #     max_iter = 20
    # )


    # x_opt, trayectoria = NewtonMethod(
    #     G=sphere.Df,
    #     H=sphere.H,
    #     x0=x0_sphere,
    # )

    x_opt, trayectoria = MDG_DM(
        f=sphere.f,
        Df=sphere.Df,
        x0=x0_sphere,
    )

    # lim debe estar dentro del dominio de la función
    plotter = FunctionPlotter2D(lim=10, n=200)
    plotter.plot(
        f=sphere.f,
        title="Función",
        minimo=(0, 0),
        trayectoria=trayectoria
    )