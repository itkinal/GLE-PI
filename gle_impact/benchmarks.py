"""Independent analytical/reduced-dimensional checks."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from .model import Model, Variant


def quadratic_impact(Q, T, L0=1.0, omega=0.5):
    return np.sign(Q)*np.sqrt(abs(Q)/(omega*T))*np.tanh(np.sqrt(omega*abs(Q)*T)/L0)


def fresh_ode(parameters, schedule, times, initial_D=0.0):
    model = Model(parameters, Variant.FRESH)
    times = np.asarray(times)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Times must increase strictly from zero")
    H = times[-1]
    boundaries = sorted({0.0, H, *(b for b in schedule.breaks if 0 < b < H)})
    result = np.empty_like(times)
    result[0], D = initial_D, initial_D
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        # Endpoint evaluations use the rate on the interval being integrated.
        def rhs(t, d):
            q = schedule.rate(np.clip(t, np.nextafter(a,b), np.nextafter(b,a)))
            return (q-model.response(d))/parameters.L0
        solution = solve_ivp(rhs, (a,b), [D], method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True)
        if not solution.success: raise RuntimeError(solution.message)
        mask = (times > a) & (times <= b)
        result[mask] = solution.sol(times[mask])[0]
        D = solution.y[0,-1]
    return result


def linear_stationary_covariance(parameters):
    """No-order quadratic latent block (Y,h); g=0. No Gibbs assumption."""
    p = parameters
    n = len(p.a)
    A = np.zeros((n+1,n+1))
    A[0,0], A[0,1:] = -p.u2, -np.array(p.a)
    A[1:,:] = A[0,:]
    A[1:,1:] -= np.diag(p.gamma)
    if max(np.linalg.eigvals(A).real) >= 0:
        raise ValueError("Unstable linearized drift")
    B = np.zeros((n+1,n))
    B[1:,:] = -np.diag(p.noise)
    return solve_continuous_lyapunov(A, -(B@B.T))
