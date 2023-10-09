import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from scipy.sparse.linalg import spsolve



x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver with manufactured solution.

        Parameters
        ----------
        L : float
            Length of the domain in both x and y directions.
        ue : sympy function
            The analytical solution used for manufactured solutions.
        """
        self.L = L
        self.ue = ue
        self.f_sympy = sp.diff(self.ue, x, 2) + sp.diff(self.ue, y, 2)
        self.f = sp.lambdify((x, y), self.f_sympy, 'numpy')
        self.ue = sp.lambdify((x, y), self.ue, 'numpy')


    def create_mesh(self, N):
        """Create a 2D mesh grid.

        Parameters
        ----------
        N : int
            Number of grid points in each dimension.
        
        Returns
        -------
        xij, yij : ndarray, ndarray
            Meshgrid for x and y coordinates.
        """
        self.N = N
        self.h = self.L / N
        x = np.linspace(0, self.L, N + 1)
        y = np.linspace(0, self.L, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')
        self.f_values = self.f(self.xij, self.yij)
        self.ue_values = self.ue(self.xij, self.yij)
        return self.xij, self.yij

    def D2(self):
        """Construct and return the second-order finite difference matrix D.
    
        D is a 1D Laplacian matrix that is used in constructing the 2D Laplacian.
        
        Returns
        -------
        D : scipy.sparse.lil_matrix
            The second-order finite difference matrix.
        """
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D
    

    def laplace(self):
        """Construct and return the 2D Laplacian matrix.
        
        This is a block matrix constructed from the 1D Laplacian matrix D.
        
        Returns
        -------
        A : scipy.sparse.csr_matrix
            The 2D Laplacian matrix.
        """
        D2 = (1. / self.h ** 2) * self.D2()
        return (sparse.kron(D2, sparse.eye(self.N + 1)) + sparse.kron(sparse.eye(self.N + 1), D2))
    

    def get_boundary_indices(self):
        """Get the indices of the boundary points in the flattened solution array.
        
        Returns
        -------
        bnds : ndarray
            Array containing the indices of the boundary points.
        """
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        return bnds


    def assemble(self):
        """Assemble the system matrix A and the right-hand side vector b.
        
        This takes into account the boundary conditions by modifying
        the rows of A corresponding to boundary points.
        
        Returns
        -------
        A : scipy.sparse.csr_matrix
            The system matrix.
        b : ndarray
            The right-hand side vector.
        """
        A = self.laplace()
        A = A.tolil()
        bnds = self.get_boundary_indices()
        for i in bnds:
            A[i, :] = 0
            A[i, i] = 1
        A = A.tocsr()

        b = self.f_values.ravel()
        b[bnds] = self.ue_values.ravel()[bnds]

        return A, b


    def l2_error(self, u):
        """Compute the L2 error norm between the numerical and exact solutions.
    
        Parameters
        ----------
        u : ndarray
            The numerical solution.
            
        Returns
        -------
        float
            The L2 error norm.
        """
        return np.sqrt(self.h ** 2 * np.sum((u - self.ue_values) ** 2))
    

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = spsolve(A, b).reshape((N + 1, N + 1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)
    


    def eval(self, x, y):
        """Evaluate the solution u(x, y) at a point.

        Parameters
        ----------
        x, y : float
            Coordinates at which to evaluate the function.

        Returns
        -------
        float
            The value of u(x, y).
        """
        # Generate an interpolating function from the numerical solution
        interpolating_function = RegularGridInterpolator((self.xij[:, 0], self.yij[0, :]), self.U)
        
        # Evaluate the function at the given x, y
        return interpolating_function(np.array([[x, y]]))[0]

def test_convergence_poisson2d():
    """Test for convergence of the Poisson2D class.

    This function creates an instance of the Poisson2D class,
    computes the convergence rates, and asserts that the rate 
    of convergence is close to 2.
    """
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    """Test the interpolation accuracy of the Poisson2D class.

    This function asserts that the interpolated values are close
    to the actual values of the analytical solution.
    """
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3

# Example usage
if __name__ == '__main__':
    L = 1.0
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y)) # Example analytical solution

    poisson_solver = Poisson2D(L, ue)
    N = 100  # Number of intervals in each direction
    u = poisson_solver(N)
    plt.imshow(u, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.colorbar()
    plt.title(f'Numerical Solution of the Poisson equation\n Dirichlet bc = 0 and $ue(x, y) = {sp.latex(ue)}$ and N = {N}')
    plt.show()

    test_convergence_poisson2d()
    test_interpolation()