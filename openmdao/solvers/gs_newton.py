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
        self.newton.options['solve_subsystems'] = False   
        self.newton.doing_hybrid = True     

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
        Gauss-Siedel and Netwon's Method.

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

        # Flag used later if need to recheck Newton mid nlgs
        self.nlgs.newton_recheck = False

        # Set the number of remaining iterations for each solver type
        self.nlgs_maxiter = self.options['maxiter_nlgs']
        self.newton_maxiter = self.options['maxiter_newton']

        # First perform 4 iterations of NLGS and time them
        # 1 iteration is required for setup, the other 3 to get
        # Aitken's relaxation going
        self.nlgs.options['maxiter'] = 4
        t1 = time.time()
        self.nlgs.solve(params, unknowns, resids, system, metadata)
        t2 = time.time()
        
        # Obtain an average time per iteration
        gs_time = (t2 - t1) / self.nlgs.options['maxiter']
        print("gs_time", gs_time)
        
        self.nlgs_maxiter -= self.nlgs.options['maxiter'] # update iteration limit
        
        # No need to go further if converged 
        if resids.norm() < self.options['atol']:
            return
        
        # This contains the residual values from self.nlgs.solve()
        # It is a list to which residual norm vals are appended
        gs_rr = self.nlgs.resids_record 
        
        # Create a cache of the unknowns if required for switching back to nlgs
        unknowns_cache = np.zeros(unknowns.vec.shape)
        unknowns_cache[:] = unknowns.vec
        
        # Check if nlgs diverged after the initial iterations
        gs_initial_diverge = False
        if gs_rr[-2] < gs_rr[-1]: # IMPROVE THIS !!!!!!!!!!!!!!!
            gs_initial_diverge = True
            
            # If nlgs initially diverged, go back to initial unknowns
            unknowns.vec[:] = self.nlgs.initial_unknowns
            gs_rate = 0.


        # Do 1 iteration of Newton and time it
        self.newton.options['maxiter'] = 1
        t3 = time.time()
        self.newton.solve(params, unknowns, resids, system, metadata)
        t4 = time.time()        
        newton_time = t4 - t3
        print("newton_time", newton_time)
        
        # This contains the residual values from self.newton.solve()
        newton_rr = self.newton.resids_record 
        
        self.newton_maxiter -= 1 # update the iteration limit

        # Predict order of magnitude reduction of resids per unit time for nlgs
        if gs_initial_diverge == False:
            print("GS converging")
            
            # Take the average convergence const of the last 2 iterations
            # !!!!!!!!!!!! WHAT IF THERE WAS DIVERGENCE THEN convergence ???
            gs_conv_const = (gs_rr[-1] / gs_rr[-2] + gs_rr[-2] / gs_rr[-3]) / 2
            
            gs_rate = - np.log10(gs_conv_const) / gs_time
            
            print("predicted GS reduction per time:", gs_rate)
        
        # Check if Newton converging and predict reduction per time
        if newton_rr[-1] < newton_rr[-2]:
            print("Newton converging")
            
            newton_conv_const = newton_rr[-1] / newton_rr[-2]
                        
            newton_rate = - np.log10(newton_conv_const) / newton_time
            
            print("predicted Newton reduction per time:", newton_rate)
            
        else:
            print("Newton diverging")
            self.nlgs.newton_diverging = True
            newton_rate = 0.

        # Switch back to nlgs if following is True
        if resids.norm() > self.options['atol'] and \
            (gs_rate > newton_rate or self.nlgs.newton_diverging): 
            # Also continues with GS if both are diverging
        
            print("SWITCHING TO NLBGS")
            
            # !!!! NEED TO THINK ABOUT CONTINUING THE RELAXATION
            
            # If Newton diverged then reset unknowns
            if self.nlgs.newton_diverging == True:
                unknowns.vec[:] = unknowns_cache 
                # Should this be done even if Newton isn't diverging?
            
            self.nlgs.options['maxiter'] = self.nlgs_maxiter
            self.nlgs.newton_ref_time = newton_time
            self.nlgs.solve(params, unknowns, resids, system, metadata)
            
            unknowns_cache[:] = unknowns.vec
            self.nlgs_maxiter -= self.nlgs.iter_count # update iteration limit
        
        # This point is reached if Newton is initially better or if the Newton 
        # performance is being rechecked in between nlgs iterations    
        while resids.norm() > self.options['atol'] and self.nlgs_maxiter > 0 \
            and self.newton_maxiter > 0:
            
            if self.nlgs.newton_recheck == False:
            
                print("SWITCHING TO NEWTON")
                self.newton.options['maxiter'] = self.newton_maxiter
                self.newton.solve(params, unknowns, resids, system, metadata)
                self.nlgs.newton_diverging = True 
                
                self.newton_maxiter -= self.newton.iter_count # update the iteration limit     

            else:
                
                print("SWITCHING TO NEWTON for 1 iteration rate check")
                self.newton.options['maxiter'] = 1
                t3 = time.time()
                self.newton.solve(params, unknowns, resids, system, metadata)
                t4 = time.time()        
                newton_time = t4 - t3
                
                self.newton_maxiter -= 1 # update the iteration limit

                # Is Newton converging? Recheck relative performance
                
                newton_rr = self.newton.resids_record # this contains the residual values
                if newton_rr[-1] < newton_rr[-2]:
                    print("Newton converging after recheck")
                    
                    newton_conv_const = newton_rr[-1] / newton_rr[-2]
                    
                    newton_rate = - np.log10(newton_conv_const) / newton_time
                    
                    self.nlgs.newton_diverging = False
                    
                    gs_rr = self.nlgs.resids_record # this contains the residual values
                    if gs_rr[-1] < gs_rr[-2]:
                        
                        gs_conv_const = (gs_rr[-1] / gs_rr[-2] + gs_rr[-2] / gs_rr[-3]) / 2
                        # using average of rate constants from 2 iteration
                                    
                        gs_rate = - np.log10(gs_conv_const) / gs_time
                        
                    else:
                        gs_rate = 0
                
                else:
                    print("Newton diverging")  
                    self.nlgs.newton_diverging = True
                    newton_rate = 0


            if resids.norm() > self.options['atol']:
                
                if gs_rate > newton_rate or self.nlgs.newton_diverging:
                
                    if self.nlgs.newton_recheck == True and self.nlgs.newton_diverging:
                        unknowns.vec[:] = unknowns_cache
                    
                    self.nlgs.options['maxiter'] = self.nlgs_maxiter
                    self.nlgs.newton_ref_time = self.newton.newton_time
                    self.nlgs.solve(params, unknowns, resids, system, metadata)
                    unknowns_cache[:] = unknowns.vec
                    self.nlgs_maxiter -= self.nlgs.iter_count # update the iteration limit
                    
                else:
                    self.nlgs.newton_recheck = False

