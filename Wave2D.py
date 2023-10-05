import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.pyplot as plt
from sympy import lambdify


x, y, t = sp.symbols('x,y,t')

class Wave2D:
    def __init__(self):
        self.L = 1.0  # Assuming domain is [0, 1] x [0, 1]

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = self.L / N  # Adjusting for the domain [0, 1]
        x = np.linspace(0, self.L, N + 1)
        y = np.linspace(0, self.L, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse=sparse)
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D / self.h ** 2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        # Assuming self.mx and self.my are the wave numbers in x and y directions
        return self.c**2 * (self.mx**2 + self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*(-y))*sp.cos(self.w*t)
    
    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c
    
    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.create_mesh(N)
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, N+1, N+1))

        ue_symbolic = self.ue(mx, my)
        ue_func = lambdify((x, y, t), ue_symbolic, 'numpy')
        ue_evaluated = ue_func(self.xij, self.yij, 0)
        self.Unm1[:] = ue_evaluated

        D = self.D2(N)
        self.Un[:] = self.Unm1[:] + 0.5 * (self.c * self.dt) ** 2 * (D @ self.Unm1 + self.Unm1 @ D.T)

    # def l2_error(self, u, t0):
    #     """Return l2-error norm
    #     Parameters
    #     ----------
    #     u : array
    #         The solution mesh function
    #     t : number
    #         The time of the comparison
    #     """
    #     ue_symbolic = self.ue(self.mx, self.my)
    #     ue_func = lambdify((x, y, t), ue_symbolic, 'numpy')
    #     ue_evaluated = ue_func(self.xij, self.yij, t0)

    #     err = np.sqrt(self.h ** 2 * np.sum((u - ue_evaluated) ** 2))
    #     return err
    def l2_error(self, u, t0, diagnostic_plot=False):
        """Return l2-error norm
        Parameters
        ----------
        u : array
            The solution mesh function
        t : number
            The time of the comparison
        diagnostic_steps : list
            List of time steps at which to plot the diagnostic figures
        """
        ue_symbolic = self.ue(self.mx, self.my)
        ue_func = lambdify((x, y, t), ue_symbolic, 'numpy')
        ue_evaluated = ue_func(self.xij, self.yij, t0)
        
        err = np.sqrt(self.h ** 2 * np.sum((u - ue_evaluated) ** 2))
        
        if diagnostic_plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Find global vmin and vmax
            global_vmin = min(u.min(), ue_evaluated.min())
            global_vmax = max(u.max(), ue_evaluated.max())

            # Plotting with the same color scale
            im1 = axs[0].imshow(u, origin='lower', extent=[0, self.L, 0, self.L], cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axs[0].set_title(f'Numerical Solution at t={t0}')

            im2 = axs[1].imshow(ue_evaluated, origin='lower', extent=[0, self.L, 0, self.L], cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axs[1].set_title(f'Analytical Solution at t={t0}')

            # Single colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im1, cax=cbar_ax)

            fig.suptitle(f'l2 error = {err}')
            plt.show()


            pass
        
        return err


    def apply_bcs(self):
        """Apply boundary conditions"""
        self.Unp1[0, :] = 0
        self.Unp1[-1, :] = 0
        self.Unp1[:, 0] = 0
        self.Unp1[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        
        # Initialize
        self.initialize(N, mx, my)
        dt = self.dt

        # For storing data
        if store_data > 0:
            stored_data = {0: self.Unm1.copy()}

        D = self.D2(N).toarray()  # Convert to dense for simplicity here

        errors = []

        # Time stepping
        for n in range(1, Nt+1):
            # Compute the next time step
            self.Unp1 = 2 * self.Un - self.Unm1 + (self.c * dt) ** 2 * (D @ self.Un + self.Un @ D.T)
            
            # Apply boundary conditions
            self.apply_bcs()

            # Shift variables for the next time step
            self.Unm1, self.Un = self.Un, self.Unp1

            # err = self.l2_error(self.Unp1, n * dt)
            err = self.l2_error(self.Unp1, n*dt)


            errors.append(err)

            if n == Nt:
                self.l2_error(self.Unp1, n*dt, True)

            # Store data if required
            if store_data > 0 and n % store_data == 0:
                stored_data[n] = self.Unp1.copy()

        if store_data > 0:
            return stored_data
        else:
            # err = self.l2_error(self.Unp1, Nt * dt)
            return self.h, errors
        
        
    def convergence_rates(self, m=4, cfl=0.05, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            plt.figure()
            t_values = np.array(range(1, Nt + 1))
            plt.plot(t_values*self.dt, err, marker='o')
            plt.xlabel('Time Step')
            plt.ylabel('L2 Error')
            plt.title('L2 Error at Each Time Step')
            plt.show()
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m+1, 1)]
        print('E:', E,'\n')
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = super().D2(N).toarray()  # Call the parent class's D2 and convert to dense array
        D[0, :3] = [-2, 2, 0]
        D[-1, -3:] = [0, 2, -2]
        return sparse.csr_matrix(D)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        self.Unp1[0, :] = self.Unp1[1, :]
        self.Unp1[-1, :] = self.Unp1[-2, :]
        self.Unp1[:, 0] = self.Unp1[:, 1]
        self.Unp1[:, -1] = self.Unp1[:, -2]

def plot(xij, yij, data):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    z_min = np.min([np.min(matrix) for matrix in data.values()])
    z_max = np.max([np.max(matrix) for matrix in data.values()])

    # def init_surface():
    #     surf = ax.plot_surface(xij, yij, data[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #     return surf,

    # def animate_surface(i):
    #     ax.clear()
    #     surf = ax.plot_surface(xij, yij, data[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #     ax.set_zlim([z_min, z_max])  # set the z-axis limits here
    #     return surf,

    def init_wireframe():
        wire = ax.plot_wireframe(xij, yij, data[0], cmap=cm.coolwarm, linewidth=0.5)
        return wire,

    def animate_wireframe(i):
        ax.clear()
        wire = ax.plot_wireframe(xij, yij, data[i], cmap=cm.coolwarm, linewidth=0.5)
        ax.set_zlim([z_min, z_max])  # set the z-axis limits here
        return wire,


    total_time = 10  # total time for the animation in seconds
    frames = range(0, len(data), 1)  # Number of frames in the animation
    FPS = int(len(frames) / total_time)  # Calculate FPS based on total time and number of frames
    interval = int(1000 / FPS)  # Interval in milliseconds

    ani = animation.FuncAnimation(fig, animate_wireframe, frames=frames, init_func=init_wireframe, interval=interval, blit=True, repeat_delay=1000)
    # ani = animation.FuncAnimation(fig, animate_surface, frames=frames, init_func=init_surface, interval=interval, blit=True, repeat_delay=1000)

    ani.save('wavemovie2d.gif', writer='pillow', fps=FPS)



def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print(E)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    raise NotImplementedError

if __name__=='__main__':
    test_convergence_wave2d()
    # test_convergence_wave2d_neumann()


    # Create an instance and solve
    wave2d_solver = Wave2D()
    N = 100 # The number of uniform intervals in each direction
    Nt = 250 # Number of time steps

    print('Starting to solve the 2d wave equation.')
    data = wave2d_solver(N, Nt, cfl=0.05, c=1, mx=3, my=3, store_data=1)
    print('Finished solving the 2d wave equation.')

    # Extract the mesh grids for plotting
    xij = wave2d_solver.xij
    yij = wave2d_solver.yij

    print('Making the animation of the solution.')
    plot(xij, yij, data)

