""" Gauss Seidel non-linear solver."""

from math import isnan

import numpy as np

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.util.record_util import update_local_meta, create_local_meta


class NLGaussSeidel(NonLinearSolver):
    """ Nonlinear Gauss Seidel solver. This is the default solver for a
    `Group`. If there are no cycles, then the system will solve its
    subsystems once and terminate. Equivalent to fixed point iteration in
    cases with cycles.

    Options
    -------
    options['alpha'] :  float(1.0)
        Relaxation factor.
    options['atol'] :  float(1e-06)
        Absolute convergence tolerance.
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print iteration totals to
        stdout, set to 2 to print the residual each iteration to stdout.
    options['maxiter'] :  int(100)
        Maximum number of iterations.
    options['rtol'] :  float(1e-06)
        Relative convergence tolerance.
    options['utol'] :  float(1e-12)
        Convergence tolerance on the change in the unknowns.

    """

    def __init__(self):
        super(NLGaussSeidel, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-6, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-6, lower=0.0,
                       desc='Relative convergence tolerance.')
        opt.add_option('utol', 1e-12, lower=0.0,
                       desc='Convergence tolerance on the change in the unknowns.')
        opt.add_option('maxiter', 100, lower=0,
                       desc='Maximum number of iterations.')
        opt.add_option('alpha', 1.0,
                       desc='Over-relaxation factor.') # Shamsheer Added

        self.print_name = 'NLN_GS'
        self.delta_n_1 = "None" # Shamsheer Added
        self.aitken_alfa = .75 # Shamsheer Added

    def setup(self, sub):
        """ Initialize this solver.

        Args
        ----
        sub: `System`
            System that owns this solver.
        """
        if sub.is_active():
            self.unknowns_cache = np.empty(sub.unknowns.vec.shape)

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system using Gauss Seidel.

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

        atol = self.options['atol']
        rtol = self.options['rtol']
        utol = self.options['utol']
        maxiter = self.options['maxiter']
        iprint = self.options['iprint']
        alpha = self.options['alpha']
        unknowns_cache = self.unknowns_cache

        # Initial run
        self.iter_count = 1

        # Metadata setup
        local_meta = create_local_meta(metadata, system.pathname)
        system.ln_solver.local_meta = local_meta
        update_local_meta(local_meta, (self.iter_count,))

        # Initial Solve
        system.children_solve_nonlinear(local_meta)

        self.recorders.record_iteration(system, local_meta)

        # Bail early if the user wants to.
        if maxiter == 1:
            return

        resids = system.resids
        unknowns_cache = np.zeros(unknowns.vec.shape)
        unknowns_cache[:] = unknowns.vec


        # Evaluate Norm
        system.apply_nonlinear(params, unknowns, resids)
        normval = resids.norm()
        basenorm = normval if normval > atol else 1.0
        u_norm = 1.0e99

        if self.options['iprint'] == 2:
            self.print_norm(self.print_name, system.pathname, 1, normval, basenorm)

        while self.iter_count < maxiter and \
                normval > atol and \
                normval/basenorm > rtol  and \
                u_norm > utol:

            # Metadata update
            self.iter_count += 1
            update_local_meta(local_meta, (self.iter_count,))

            ################################################################
            # Start of code added by Shamsheer Chauhan
            ################################################################

            use_acc = True

            if use_acc:
                unknowns_cache[:] = unknowns.vec

                # Runs an iteration
                system.children_solve_nonlinear(local_meta)
                self.recorders.record_iteration(system, local_meta)

                # print("new unknowns vec is", unknowns.vec)

                if type(self.delta_n_1) is not str:

                    # Method 1 used by kenway et al.
                    delta_n = unknowns.vec - unknowns_cache
                    print("delta_n norm is ", np.linalg.norm(delta_n))
                    delta_n_1 = self.delta_n_1
                    print("delta_n_1 norm is ", np.linalg.norm(delta_n_1))
                    self.aitken_alfa = self.aitken_alfa * (1. - np.dot(( delta_n  - delta_n_1), delta_n) / np.linalg.norm(( delta_n  - delta_n_1), 2)**2)
                    self.aitken_alfa = max(0.5, min(1.25, self.aitken_alfa))

                    print("Aitken alfa is", self.aitken_alfa)

                    self.delta_n_1 = delta_n.copy()
                    unknowns.vec[:] = unknowns_cache + self.aitken_alfa * delta_n
                    # print("relaxed unknowns vec is", unknowns.vec)

                    # Simple relaxation
                    # unknowns.vec[:] = (1-alpha)*unknowns_cache + alpha*unknowns.vec # Relaxation
                    # unknowns_cache[:] = unknowns.vec

                else:
                    # print("First iter")
                    self.delta_n_1 = unknowns.vec - unknowns_cache # Method 2 used by kenway et al.


            ################################################################
            # End of code added by Shamsheer Chauhan
            ################################################################

            else:
                # Runs an iteration
                system.children_solve_nonlinear(local_meta)
                self.recorders.record_iteration(system, local_meta)


            # Evaluate Norm
            system.apply_nonlinear(params, unknowns, resids)
            normval = resids.norm()
            u_norm = np.linalg.norm(unknowns.vec - unknowns_cache)

            u_norm = 100 ### !!!!!!!!! Shamsheer HARD CODED u_norm !!!!!!!!!!!

            if self.options['iprint'] == 2:
                self.print_norm(self.print_name, system.pathname, self.iter_count, normval,
                                basenorm, u_norm=u_norm)

        # Final residual print if you only want the last one
        if self.options['iprint'] == 1:
            self.print_norm(self.print_name, system.pathname, self.iter_count, normval,
                            basenorm, u_norm=u_norm)

        if self.iter_count >= maxiter or isnan(normval):
            msg = 'FAILED to converge after %d iterations' % self.iter_count
            fail = True
        else:
            fail = False

        if self.options['iprint'] > 0 or fail:
            if not fail:
                msg = 'Converged in %d iterations' % self.iter_count

            self.print_norm(self.print_name, system.pathname, self.iter_count, normval,
                            basenorm, msg=msg)

        if fail and self.options['err_on_maxiter']:
            raise AnalysisError("Solve in '%s': NLGaussSeidel %s" %
                                (system.pathname, msg))
