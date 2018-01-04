from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.solvers.newton import Newton
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel

import numpy as np
import time

class GSthenNewton(NonLinearSolver):

    def __init__(self):
        super(GSthenNewton, self).__init__()
        
        opt = self.options
        opt.add_option('atol', 1e-6, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-6, lower=0.0,
                       desc='Relative convergence tolerance.')
        opt.add_option('utol', 1e-12, lower=0.0,
                       desc='Convergence tolerance on the change in the unknowns.')
        opt.add_option('maxiter_nlgs', 10, lower=0,
                       desc='Maximum number of iterations.')
        opt.add_option('maxiter_newton', 15, lower=0,
                       desc='Maximum number of iterations.')

        self.nlgs = NLGaussSeidel()

        self.newton = Newton()    

    def setup(self, sub):
        """ Initialize sub solvers.

        Args
        ----
        sub: `System`
            System that owns this solver.
        """
        self.nlgs.setup(sub)
        self.newton.setup(sub)
        
        # Set the specified print levels for the solvers
        self.newton.options['iprint'] = self.options['iprint']
        self.nlgs.options['iprint'] = self.options['iprint']
        
        # Set the specified tolerances for the solvers
        self.nlgs.options['atol'] = self.options['atol']
        self.nlgs.options['rtol'] = self.options['rtol']
        self.nlgs.options['utol'] = self.options['utol']
        
        self.newton.options['atol'] = self.options['atol']
        self.newton.options['rtol'] = self.options['rtol']
        self.newton.options['utol'] = self.options['utol']

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system by first doing a few iterations of NLBGS and then 
        switching to Netwon's Method. This was created for the additional
        comparison for the automated selection paper.
        

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        # NLBGS
        self.nlgs.options['maxiter'] = self.options['maxiter_nlgs']
        self.nlgs.solve(params, unknowns, resids, system, metadata)

        # Newton
        self.newton.options['maxiter'] = self.options['maxiter_newton']
        self.newton.solve(params, unknowns, resids, system, metadata)
