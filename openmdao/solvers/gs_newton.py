from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.solvers.newton import Newton
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel

import numpy as np
import time

class HybridGSNewton(NonLinearSolver):

    def __init__(self):
        super(HybridGSNewton, self).__init__()
        
        opt = self.options
        opt.add_option('atol', 1e-6, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-6, lower=0.0,
                       desc='Relative convergence tolerance.')
        opt.add_option('utol', 1e-12, lower=0.0,
                       desc='Convergence tolerance on the change in the unknowns.')
        opt.add_option('maxiter_nlgs', 1000, lower=0,
                       desc='Maximum number of iterations.')
        opt.add_option('maxiter_newton', 20, lower=0,
                       desc='Maximum number of iterations.')

        self.nlgs = NLGaussSeidel()
        self.nlgs.doing_hybrid = True

        self.newton = Newton()
        # self.newton.options['solve_subsystems'] = False   
        self.newton.doing_hybrid = True     
        
        self.status = 'None'

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
        """ Solves the system by selecting and switching between nonlinear 
        Gauss-Siedel and Netwon's Method. This was used for the automated
        selection paper.

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

        self.nlgs.newton_diverging = False # Flag used if Newton diverges
        self.nlgs.newton_recheck = False # Flag used later if need to recheck Newton mid nlgs
        stalled = False # Flag used to exit if convergence stalls

        # Set the number of remaining iterations for each solver type
        self.nlgs_maxiter = self.options['maxiter_nlgs']
        self.nlgs.newton_maxiter = self.options['maxiter_newton']
        self.newton.nlgs_maxiter = self.nlgs_maxiter
        self.nlbgs_run_out_flag = False

        # First perform 4 iterations of NLGS and time them
        # 1 iteration for setup, the other 3 to get relaxation going
        self.nlgs.options['maxiter'] = 4
        t1 = time.time()
        self.nlgs.solve(params, unknowns, resids, system, metadata)
        t2 = time.time()
        
        # No need to go further if converged 
        if resids.norm() < self.options['atol']:
            self.status = 'Successful'
            return
        
        # Obtain an average time per iteration
        gs_time = (t2 - t1) / self.nlgs.options['maxiter']; print("gs_time", gs_time)
        # update iteration limit
        self.nlgs_maxiter -= self.nlgs.options['maxiter'] 
        
        # This contains the residual values from self.nlgs.solve()
        # It is a list to which residual norm vals are appended
        gs_rr = self.nlgs.resids_record 
        
        gs_conv_const = (gs_rr[-1] / gs_rr[-2] + gs_rr[-2] / gs_rr[-3]) / 2
        gs_rate = - np.log10(gs_conv_const) / gs_time
        print("predicted GS reduction per time:", gs_rate)
        
        # Check if nlgs diverged after the initial iterations
        if gs_rate < 0:
            # Go back to initial unknowns
            unknowns.vec[:] = self.nlgs.initial_unknowns
            
        # Create a cache of the unknowns if required for switching back to nlgs
        unknowns_cache = np.zeros(unknowns.vec.shape)
        unknowns_cache[:] = unknowns.vec

        # Do 1 iteration of Newton and time it
        self.newton.options['maxiter'] = 1
        t3 = time.time()
        self.newton.solve(params, unknowns, resids, system, metadata)
        t4 = time.time()        
        newton_time = t4 - t3
        print("newton_time", newton_time)
        self.nlgs.newton_maxiter -= 1 # update the iteration limit
        
        # This contains the residual values from self.newton.solve()
        newton_rr = self.newton.resids_record 
        
        # Predict reduction per time
        newton_conv_const = newton_rr[-1] / newton_rr[-2]
        newton_rate = - np.log10(newton_conv_const) / newton_time
        print("predicted Newton reduction per time:", newton_rate)

        if newton_rate < 0:
            print("Newton diverging")
            self.nlgs.newton_diverging = True
            unknowns.vec[:] = unknowns_cache
        elif newton_rate < gs_rate:
            self.nlgs.newton_worked = np.zeros(unknowns.vec.shape)
            self.nlgs.newton_worked[:] = unknowns.vec
            
        while resids.norm() > self.options['atol'] and (self.nlgs_maxiter > 0 or self.nlgs.newton_maxiter > 0)\
        and stalled == False:
            
            if (gs_rate > newton_rate or self.nlgs.newton_diverging) and self.nlgs_maxiter > 0:
                
                print("SWITCHING TO NLBGS")
            
                if self.nlgs.newton_recheck == True and self.nlgs.newton_diverging:
                    unknowns.vec[:] = unknowns_cache
                
                self.nlgs.options['maxiter'] = self.nlgs_maxiter
                self.nlgs.newton_ref_time = self.newton.newton_time
                t1 = time.time()
                self.nlgs.solve(params, unknowns, resids, system, metadata)
                t2 = time.time()
                unknowns_cache[:] = unknowns.vec
                self.nlgs_maxiter -= self.nlgs.iter_count # update the iteration limit
                gs_time = (t2 - t1) / self.nlgs.iter_count
                gs_rr = self.nlgs.resids_record # this contains the residual values
                gs_conv_const = (gs_rr[-1] / gs_rr[-2] + gs_rr[-2] / gs_rr[-3]) / 2
                gs_rate = - np.log10(gs_conv_const) / gs_time
                
                if self.nlgs.stall_flag == True and resids.norm() < 10*self.options['atol']:
                    stalled = True
                
            else:
                self.nlgs.newton_recheck = False
                
            if resids.norm() < self.options['atol'] or (self.nlgs_maxiter <= 0 and self.nlgs.newton_maxiter <= 0)\
            or stalled == True:
                if resids.norm() < self.options['atol']:
                    self.status = 'Successful'
                else:
                    self.status = 'Failed'
                return
            
            if self.nlgs_maxiter < 1 and self.nlbgs_run_out_flag == False:
                self.nlbgs_run_out_flag = True
                unknowns.vec[:] = self.nlgs.unknowns_cache_copy
            
            if (self.nlgs.newton_recheck == False or self.nlgs.stall_flag == True) and self.nlgs.newton_maxiter > 0:
            
                print("SWITCHING TO NEWTON")
                self.nlgs.checked_once_while_diverging = False
                self.newton.nlgs_maxiter = self.nlgs_maxiter
                self.newton.options['maxiter'] = self.nlgs.newton_maxiter
                self.newton.solve(params, unknowns, resids, system, metadata)
                
                if self.newton.stall_flag == True:
                    stalled = True
                
                self.nlgs.newton_diverging = True 
                
                self.nlgs.newton_maxiter -= self.newton.iter_count # update the iteration limit
                
                newton_rr = self.newton.resids_record # this contains the residual values
                
                # Exit if Newton is hopeless and we are out of NLBGS iterations
                if newton_rr[-1] > 1e14*self.options['atol'] and self.nlgs_maxiter < 1:
                    if resids.norm() < self.options['atol']:
                        self.status = 'Successful'
                    else:
                        self.status = 'Failed'
                    return

            elif self.nlgs.newton_maxiter > 0:
                
                print("SWITCHING TO NEWTON for 1 iteration rate check")
                self.newton.options['maxiter'] = 1
                t3 = time.time()
                self.newton.solve(params, unknowns, resids, system, metadata)
                t4 = time.time()        
                newton_time = t4 - t3
                
                self.nlgs.newton_maxiter -= 1 # update the iteration limit

                # Is Newton converging? Recheck relative performance
                newton_rr = self.newton.resids_record # this contains the residual values
                newton_conv_const = newton_rr[-1] / newton_rr[-2]
                newton_rate = - np.log10(newton_conv_const) / newton_time
                
                if newton_rate > 0:
                    print("Newton converging after recheck")
                    self.nlgs.newton_diverging = False
                
                else:
                    print("Newton diverging after recheck")  
                    self.nlgs.newton_diverging = True
        
        if resids.norm() < self.options['atol']:
            self.status = 'Successful'
        else:
            self.status = 'Failed'
            