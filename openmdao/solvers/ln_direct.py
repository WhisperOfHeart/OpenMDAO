""" OpenMDAO LinearSolver that explicitly solves the linear system using
linalg.solve or scipy LU factor/solve. Inherits from MultLinearSolver just
for the mult function."""

from collections import OrderedDict

import numpy as np
from scipy.linalg import lu_factor, lu_solve
import scipy.sparse # Shamsheer added
import scipy.sparse.linalg
import time

from openmdao.solvers.solver_base import MultLinearSolver


class DirectSolver(MultLinearSolver):
    """ OpenMDAO LinearSolver that explicitly solves the linear system using
    linalg.solve. The user can choose to have the jacobian assembled
    directly or through matrix-vector product.

    Options
    -------
    options['iprint'] :  int(0)
        Set to 0 to print only failures, set to 1 to print iteration totals to
        stdout, set to 2 to print the residual each iteration to stdout,
        or -1 to suppress all printing.
    options['mode'] :  str('auto')
        Derivative calculation mode, set to 'fwd' for forward mode, 'rev' for
        reverse mode, or 'auto' to let OpenMDAO determine the best mode.
    options['jacobian_method'] : str('MVP')
        Method to assemble the jacobian to solve. Select 'MVP' to build the
        Jacobian by calling apply_linear with columns of identity. Select
        'assemble' to build the Jacobian by taking the calculated Jacobians in
        each component and placing them directly into a clean identity matrix.
    options['solve_method'] : str('LU')
        Solution method, either 'solve' for linalg.solve, or 'LU' for
        linalg.lu_factor and linalg.lu_solve.
    """

    def __init__(self):
        super(DirectSolver, self).__init__()
        self.options.remove_option("err_on_maxiter")
        self.options.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +
                       "let OpenMDAO determine the best mode.",
                       lock_on_setup=True)

        self.options.add_option('jacobian_method', 'MVP', values=['MVP', 'assemble'],
                                desc="Method to assemble the jacobian to solve. " +
                                "Select 'MVP' to build the Jacobian by calling " +
                                "apply_linear with columns of identity. Select " +
                                "'assemble' to build the Jacobian by taking the " +
                                "calculated Jacobians in each component and placing " +
                                "them directly into a clean identity matrix.")
        self.options.add_option('solve_method', 'LU', values=['LU', 'solve'],
                                desc="Solution method, either 'solve' for linalg.solve, " +
                                "or 'LU' for linalg.lu_factor and linalg.lu_solve.")

        self.jacobian = None
        self.lup = None
        self.mode = None
        
        self.counter = 0 # Shamsheer added

    def setup(self, system):
        """ Initialization. Allocate Jacobian and set up some helpers.

        Args
        ----
        system: `System`
            System that owns this solver.
        """

        # Only need to setup if we are assembling the whole jacobian
        if self.options['jacobian_method'] == 'MVP':
            return

        # Note, we solve a slightly modified version of the unified
        # derivatives equations in OpenMDAO.
        # (dR/du) * (du/dr) = -I
        u_vec = system.unknowns
        self.jacobian = -np.eye(u_vec.vec.size)

        # Clear the index cache
        system._icache = {}

    def solve(self, rhs_mat, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        Args
        ----
        rhs_mat : dict of ndarray
            Dictionary containing one ndarry per top level quantity of
            interest. Each array contains the right-hand side for the linear
            solve.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        dict of ndarray : Solution vectors
        """

        self.system = system

        if self.mode is None:
            self.mode = mode

        sol_buf = OrderedDict()

        for voi, rhs in rhs_mat.items():
            self.voi = None

            if system._jacobian_changed:
                method = self.options['jacobian_method']

                # Must clear the jacobian if we switch modes
                if method == 'assemble' and self.mode != mode:
                    self.setup(system)
                self.mode = mode

                self.jacobian, _ = system.assemble_jacobian(mode=mode, method=method,
                                                            mult=self.mult)
                system._jacobian_changed = False

                if self.options['solve_method'] == 'LU':
                    self.lup = lu_factor(self.jacobian)

            if self.options['solve_method'] == 'LU':
                deriv = lu_solve(self.lup, rhs)
            else:
                deriv = np.linalg.solve(self.jacobian, rhs)

            self.system = None
            sol_buf[voi] = deriv

        return sol_buf
        
    #FOR COUPLING STRENGTH 
    # def solve(self, rhs_mat, system, mode):
    #     """ Solves the linear system for the problem in self.system. The
    #     full solution vector is returned.
    # 
    #     Args
    #     ----
    #     rhs_mat : dict of ndarray
    #         Dictionary containing one ndarry per top level quantity of
    #         interest. Each array contains the right-hand side for the linear
    #         solve.
    # 
    #     system : `System`
    #         Parent `System` object.
    # 
    #     mode : string
    #         Derivative mode, can be 'fwd' or 'rev'.
    # 
    #     Returns
    #     -------
    #     dict of ndarray : Solution vectors
    #     """
    # 
    #     self.system = system
    # 
    #     if self.mode is None:
    #         self.mode = mode
    # 
    #     sol_buf = OrderedDict()
    # 
    #     for voi, rhs in rhs_mat.items():
    #         self.voi = None
    # 
    #         if system._jacobian_changed:
    #             method = self.options['jacobian_method']
    # 
    #             # Must clear the jacobian if we switch modes
    #             if method == 'assemble' and self.mode != mode:
    #                 self.setup(system)
    #             self.mode = mode
    # 
    #             self.jacobian, _, sizelist, end_list = system.assemble_jacobian(mode=mode, method=method,
    #                                                         mult=self.mult)
    #             system._jacobian_changed = False
    #             # print(system.list_auto_order()); exit()
    # 
    #         #     if self.options['solve_method'] == 'LU':
    #         #         self.lup = lu_factor(self.jacobian)
    #         # 
    #         # if self.options['solve_method'] == 'LU':
    #         #     deriv = lu_solve(self.lup, rhs)
    #         # else:
    #         #     deriv = np.linalg.solve(self.jacobian, rhs)
    #         
    #         deriv = 1e-16 * np.ones(self.jacobian.shape[0])
    #         
    #         self.system = None
    #         sol_buf[voi] = deriv
    #     
    #     if self.counter == 0:
    #         if len(sizelist) > 1:    
    #             self.counter += 1
    #             
    #             befores = list(end_list)
    #             befores.append(0)
    #             befores = np.array(befores, dtype=int)
    #             
    #             temp_jaco = scipy.sparse.lil_matrix((self.jacobian.shape[0], self.jacobian.shape[0]))
    #             
    #             for i in range(len(end_list)):
    #                 temp_jaco[befores[i-1]:befores[i], 0:befores[i]] = self.jacobian[befores[i-1]:befores[i], 0:befores[i]]
    #                 self.jacobian[befores[i-1]:befores[i], 0:befores[i]] = 0
    #             
    #             self.jacobian = scipy.sparse.csc_matrix(self.jacobian)
    #             temp_jaco = scipy.sparse.csc_matrix(temp_jaco)
    # 
    #             A = scipy.sparse.linalg.spsolve(temp_jaco,self.jacobian)
    #             
    #             self.jacobian = None
    #             jaco = None
    #             temp_jaco = None
    #             
    #             A = A.toarray() 
    #             print("eigval is", np.max(np.abs(np.linalg.eigvals(A))))
    #             # print("eigval is", np.abs(scipy.sparse.linalg.eigs(A,k=1)[0]))
    # 
    #     else:
    #         exit()
    #     
    #     self.jacobian = None
    #     jaco = None
    #     temp_jaco = None
    # 
    #     return sol_buf
    # 
