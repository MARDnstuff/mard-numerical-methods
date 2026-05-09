import numpy as np
import matplotlib.pyplot as plt

# AI generated
class FunctionPlotter2D:
    """
    Graficador genérico para funciones de dos variables f(x1, x2).

    Genera una superficie 3D y curvas de nivel para cualquier función
    que acepte dos arrays numpy (malla) y retorne un array de valores.
    Opcionalmente superpone la trayectoria del algoritmo MDG_PF en ambas gráficas.
    """

    def __init__(self, lim: float = 3.0, n: int = 200):
        """
        :param lim: Límite del dominio [-lim, lim] en cada eje. Por defecto 3.0.
        :param n:   Resolución de la malla. Por defecto 200.
        """
        self.lim = lim
        self.n = n

    def plot(self, f: callable, title: str = "f(x1, x2)", minimo: tuple = None, levels: int = 20, trayectoria: np.ndarray = None):
        """
        Grafica la función g en superficie 3D y curvas de nivel.
        Si se proporciona una trayectoria, la superpone en ambas gráficas.

        :param g:           Función g(X, Y) -> Z que acepta arrays 2D (meshgrid).
        :param title:       Título de la gráfica. Por defecto "f(x1, x2)".
        :param minimo:      Tupla (x1, x2) del mínimo global. Si es None no se marca.
        :param levels:      Número de curvas de nivel. Por defecto 20.
        :param trayectoria: np.array shape (max_iter + 1, 2) retornado por MDG_PF.
                            trayectoria[0]  = punto inicial x0.
                            trayectoria[-1] = aproximación al mínimo.
                            Si es None no se grafica trayectoria.
        """
        g = lambda X, Y: f(np.array([X,Y]))
        x = np.linspace(-self.lim, self.lim, self.n)
        y = np.linspace(-self.lim, self.lim, self.n)
        X, Y = np.meshgrid(x, y)
        Z = g(X, Y)

        fig = plt.figure(figsize=(14, 5))
        fig.suptitle(title, fontsize=14)

        # ── Superficie 3D ──────────────────────────────────────────────
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7,
                                linewidth=0, antialiased=True)
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('$f(x)$')
        ax1.set_title('Superficie 3D')
        fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.1)

        if trayectoria is not None:
            # Evaluar f en cada punto de la trayectoria para la altura en Z
            z_tray = np.array([g(p[0], p[1]) for p in trayectoria])
            ax1.plot(trayectoria[:, 0], trayectoria[:, 1], z_tray,'-o', color='white', markersize=3,linewidth=1.5, zorder=5, label='Trayectoria')
            ax1.scatter(trayectoria[0, 0],  trayectoria[0, 1],  z_tray[0],
                        color='cyan',  s=60, zorder=6, label='Inicio $x_0$')
            ax1.scatter(trayectoria[-1, 0], trayectoria[-1, 1], z_tray[-1],
                        color='red',   s=80, marker='*', zorder=6, label='Mínimo')
            ax1.legend(fontsize=8)

        # ── Curvas de nivel ────────────────────────────────────────────
        ax2 = fig.add_subplot(122)
        cp = ax2.contourf(X, Y, Z, levels=levels, cmap='viridis')
        ax2.contour(X, Y, Z, levels=levels, colors='white',
                    linewidths=0.4, alpha=0.5)
        fig.colorbar(cp, ax=ax2)
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('$x_2$')
        ax2.set_title('Curvas de nivel')

        if minimo is not None:
            ax2.plot(*minimo, '*r', markersize=12, label=f'Mínimo global {minimo}')

        if trayectoria is not None:
            ax2.plot(trayectoria[:, 0], trayectoria[:, 1],'-o', color='white', markersize=4,linewidth=1.5, label='Trayectoria')
            ax2.plot(trayectoria[0, 0],  trayectoria[0, 1],'s', color='cyan', markersize=10, label='Inicio $x_0$')
            ax2.plot(trayectoria[-1, 0], trayectoria[-1, 1],'*', color='red',  markersize=14, label='Mínimo aproximado')

        ax2.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

