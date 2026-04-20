
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Callable, List, Tuple
from scipy.integrate._ivp import DOP853

# class of numerical solvers, each method is a static method that takes the same arguments and returns the same output format
# contains scipy solvers, my hard-coded solvers, plotting and representation functions
# can then add to this in the future

# could add fourier series function, method of frobenius, power series method, etc. 
# fast fourier transform, discrete fourier transform, etc.
# PINNS models functions????



class NumericalSolver:


    # forward euler: y_{n+1} = y_n + h * f(t_n, y_n)
    @staticmethod
    def forward_euler(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            y[i+1] = y[i] + h * np.asarray(f(t[i], y[i]))

        return y
    


    # backward euler: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    @staticmethod
    def backward_euler(f: Callable, y0: np.ndarray, t: np.ndarray, iters=5) -> np.ndarray:

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):

            h = t[i+1] - t[i]
            y_next = y[i].copy() # initial guess

            for _ in range(iters):

                y_next = y[i] + h * np.asarray(f(t[i+1], y_next))

            y[i+1] = y_next

        return y
    
    
    # midpoint method:  k_1 = y_n + (h/2) * f(t_n, y_n)
    #                   y_{n+1} = y_n + h * f(t_n + h/2, k_1)
    @staticmethod
    def midpoint_euler(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):
            
            h = t[i+1] - t[i]
            k1 = y[i] + (h/2) * np.asarray(f(t[i], y[i]))
            y[i+1] = y[i] + h * np.asarray(f(t[i] + (h/2), k1))

        return y
    
    # Heun's/trapezoidal method: y_{n+1} = y_n + (h/2) * ( f(t_n, y_n) + f(t_{n+1}, y_{n+1}) )
    @staticmethod
    def heun_euler(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):
            
            h = t[i+1] - t[i]
            k1 = np.asarray(f(t[i], y[i]))
            k2 = np.asarray(f(t[i+1], y[i] + h * k1))
            y[i+1] = y[i] + (h/2) * (k1 + k2)

        return y
    

    
    # 4th order Runge-Kutta method (RK4) - good for non-stiff problems, fixed time step
    @staticmethod
    def rk4(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):
            
            h = t[i+1] - t[i]
            ti = t[i]
            yi = y[i]
            k1 = np.asarray(f(ti, yi))
            k2 = np.asarray(f(ti + (h/2), yi + (h/2) * k1))
            k3 = np.asarray(f(ti + (h/2), yi + (h/2) * k2))
            k4 = np.asarray(f(ti + h, yi + h * k3))

            y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

        return y
    


    # adaptive RK4 with step doubling error estimation - good for non-stiff problems, variable time step
    # actual adaptive RK4 function
    @staticmethod
    def rk4_adaptive(f: Callable, y0: np.ndarray, t: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        dt = (t[1] - t[0]) / 10  # initial guess for a small step

        # internal single-step RK4 helper
        def rk4_step(f, ti, yi, h):
            k1 = np.asarray(f(ti, yi))
            k2 = np.asarray(f(ti + h/2, yi + h/2 * k1))
            k3 = np.asarray(f(ti + h/2, yi + h/2 * k2))
            k4 = np.asarray(f(ti + h, yi + h * k3))

            return yi + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

        
        for i in range(len(t) - 1):

            t_target = t[i+1]
            t_current = t[i]
            y_current = y[i].copy()
            
            # keep taking adaptive steps until we reach the next t_eval point
            while t_current < t_target:

                if t_current + dt > t_target:

                    dt = t_target - t_current # to not overstep the target
                
                # step doubling error estimation
                y_full = rk4_step(f, t_current, y_current, dt)
                y_half = rk4_step(f, t_current + dt/2, rk4_step(f, t_current, y_current, dt/2), dt/2)
                
                error = np.linalg.norm(y_full - y_half)
                
                if error <= tol or dt < 1e-12:

                    # step is accepted
                    t_current += dt
                    y_current = y_half

                    # adjusting dt for next time (bigger if error is small)
                    dt *= min(2.0, max(0.1, 0.8 * (tol / (error + 1e-15))**0.2))
                    
                else:

                    # step rejected, shrink dt and try again
                    dt *= 0.5
                    
            y[i+1] = y_current
            
        return y


    # backward differentiation order 2 (second derivative) - BDF2 - high precision for stiff problems, implicit multi-step method
    @staticmethod
    def bdf2(f: Callable, y0: np.ndarray, t: np.ndarray, iters: int = 5) -> np.ndarray:

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            
            if i == 0:

                y_next = y[i].copy()
                for _ in range(iters):
                    y_next = y[i] + h * np.asarray(f(t[i+1], y_next))
                y[i+1] = y_next

            else:

                y_next = y[i].copy() # initial guess

                for _ in range(iters):

                    # solving the implicit equation via fixed-point iteration
                    prediction = (4/3) * y[i] - (1/3) * y[i-1]
                    y_next = prediction + (2/3) * h * np.asarray(f(t[i+1], y_next))
                    
                y[i+1] = y_next

        return y
    


    @staticmethod
    def rk8_adaptive(f: Callable, y0: np.ndarray, t: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        C = DOP853.C   # length 12  — fractional time nodes
        A = DOP853.A   # 12 rows, each padded to length 12
        B = DOP853.B   # length 12  — solution weights (stages 1-12 only)
        E3 = DOP853.E3 # length 13  — error weights (uses all 13 stages)
        E5 = DOP853.E5 # length 13  — error weights (uses all 13 stages)
        
        def rk8_embedded_step(ti, yi, h):
            k = np.zeros((13, len(yi)))
            
            # stage 1
            k[0] = np.asarray(f(ti, yi))
            
            # stages 2 - 12
            for s in range(1, 12):
                dy = np.dot(A[s-1][:s], k[:s])
                k[s] = np.asarray(f(ti + C[s-1] * h, yi + h * dy))
            
            # stage 13, evaluated at ti + h using preliminary 8th-order solution
            y_inter = yi + h * np.dot(B[:12], k[:12])
            k[12] = np.asarray(f(ti + h, y_inter))
            
            # B is length 12, so only dot against the first 12 stages
            y_next = yi + h * np.dot(B, k[:12])
            
            # E3 and E5 are length 13, so use all stages for error estimation
            err5 = h * np.dot(E5, k)
            err3 = h * np.dot(E3, k)
            
            err5_sq = np.mean(err5**2)
            err3_sq = np.mean(err3**2)
            denom = err5_sq + 0.01 * err3_sq
            error = err5_sq / np.sqrt(denom) if denom > 0 else 0.0
                
            return y_next, error

        dt = (t[1] - t[0]) / 10.0
        for i in range(len(t) - 1):
            t_target, t_curr, y_curr = t[i+1], t[i], y[i].copy()
            
            while t_curr < t_target:
                if t_curr + dt > t_target:
                    dt = t_target - t_curr
                    
                y_next, error = rk8_embedded_step(t_curr, y_curr, dt)
                
                if error <= tol or dt < 1e-14:
                    t_curr += dt
                    y_curr = y_next
                    dt *= min(2.0, max(0.1, 0.9 * (tol / (error + 1e-18))**(1/9)))
                else:
                    dt *= 0.5
                    
            y[i+1] = y_curr
        return y




# this class essentially allows for a constant interface for different solvers
# means the same inputs, strucutre, outputs regardless of whether hard-coded or scipy functions
class ScipySolver:

    # RK45 (with adaptive time-stepping) - default solver
    @staticmethod
    def pythonrk45(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        sol = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, method='RK45')

        return sol.y.T
    

    # BDF (backward differentiation formula, implicit multi-step method) - good for stiff problems
    @staticmethod
    def pythonbdf(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        sol = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, method='BDF')

        return sol.y.T
    

    # Radau (implicit Runge-Kutta method, good for stiff problems) - high precision
    @staticmethod
    def pythonradau(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        sol = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, method='Radau')

        return sol.y.T
    

    # LSODA (automatically switches between non-stiff and stiff methods) - good for general use
    @staticmethod
    def pythonlsoda(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        sol = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, method='LSODA')

        return sol.y.T
    

    # DOP853 (explicit Runge-Kutta method of order 8) - high precision for non-stiff problems
    @staticmethod
    def pythondop853(f: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:

        sol = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, method='DOP853')

        return sol.y.T
    



class Results:

    @staticmethod
    def compare_solutions(results_custom, results_pro, t_steps, title_suffix_custom="Heun Euler", title_suffix_pro="RK45"):


        # plotting harmonic oscillator example
        plt.figure(figsize=(10, 5))

        # plot position (index 0)
        plt.plot(t_steps, results_custom[:, 0], 'r--', label=title_suffix_custom)
        plt.plot(t_steps, results_pro[:, 0], 'b-', label=title_suffix_pro, alpha=0.7)

        plt.title('Custom vs. Pro Solver')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 20)
        plt.show()

        # plotting harmonic oscillator example
        plt.figure(figsize=(10, 5))

        # plot velocity (index 1)
        plt.plot(t_steps, results_custom[:, 1], 'r--', label=title_suffix_custom)
        plt.plot(t_steps, results_pro[:, 1], 'b-', label=title_suffix_pro, alpha=0.7)

        plt.title('Custom vs. Pro Solver')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 20)
        plt.show()


        # phase space plot (position vs velocity) - should show a circle for a perfect harmonic oscillator, deviations indicate energy conservation issues
        plt.figure(figsize=(6, 6))
        plt.plot(results_custom[:, 0], results_custom[:, 1], 'r--', label=title_suffix_custom)
        plt.plot(results_pro[:, 0], results_pro[:, 1], 'b-', label=title_suffix_pro, alpha=0.7)
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Phase Space: Energy Conservation')
        plt.axis('equal')
        plt.legend()
        plt.show()




if __name__ == "__main__":

   # example derivative for a simple harmonic oscillator
    def harmonic_oscillator(t, y):
        pos, vel = y

        # d2xdt2 = -(k/m) * x = -w^2 * x , w = sqrt(k/m), (harmonic oscillator equation)
        # simple version: d2xdt2 = -x

        # dxdt = velocity = v
        # dvdt = acceleration = -pos = -x
        acceleration = -pos
        return [vel, acceleration]

    t_steps = np.linspace(0, 20, 100)
    y_initial = np.array([1.0, 0.0])

    # Swapping solvers is now seamless
    results_custom = NumericalSolver.heun_euler(harmonic_oscillator, y_initial, t_steps)
    results_pro    = ScipySolver.pythonrk45(harmonic_oscillator, y_initial, t_steps)

    #plotting
    Results.compare_solutions(results_custom, results_pro, t_steps, title_suffix_custom="Heun Euler", title_suffix_pro="RK45") 