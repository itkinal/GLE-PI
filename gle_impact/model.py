"""Model equations, parameter validation, and spectrum controls."""
from dataclasses import dataclass, replace
from enum import StrEnum
import numpy as np
from scipy.special import expit


class Variant(StrEnum):
    KYLE = "kyle"
    FRESH = "fresh"
    SINGLE = "single"
    GLE = "gle"


@dataclass(frozen=True)
class Parameters:
    L0: float = 1.0
    u2: float = 1.0
    u3: float = 0.0
    u4: float = 0.1
    a: tuple[float, ...] = (0.5, 0.05)
    gamma: tuple[float, ...] = (1.0, 0.1)
    c: tuple[float, ...] = (5.0, 0.5)
    lam: tuple[float, ...] = (2.0, 0.2)
    cross_b: tuple[float, ...] = (0.0, 0.0)
    cross_scale: float = 1.0
    y_rho: float = 2.0
    sigma_y: float = 1.0
    sigma_x: float = 1.0
    c_d: float = 1.0
    tau_d: float = 1.0
    q_star: float = 1.0
    pi0: float = 0.0
    response: str = "exponential"  # quadratic is a global analytical control

    def __post_init__(self):
        numeric = [self.L0, self.u2, self.u3, self.u4, self.cross_scale,
                   self.y_rho, self.sigma_y, self.sigma_x, self.c_d,
                   self.tau_d, self.q_star, self.pi0,
                   *self.a, *self.gamma, *self.c, *self.lam, *self.cross_b]
        if not np.isfinite(numeric).all():
            raise ValueError("All parameters must be finite")
        if not (len(self.a) == len(self.gamma) > 0 and
                len(self.c) == len(self.lam) == len(self.cross_b) > 0):
            raise ValueError("Mode arrays must have matching nonzero lengths")
        if min(*self.a, *self.gamma, *self.lam, self.L0, self.y_rho,
               self.cross_scale, self.sigma_x, self.c_d, self.tau_d) <= 0:
            raise ValueError("Scales, intrinsic weights, and rates must be positive")
        if min(self.sigma_y, self.q_star, self.pi0) < 0 or max(abs(np.array(self.cross_b))) >= 1:
            raise ValueError("Require nonnegative noise/counterflow and |cross_b| < 1")
        if self.u4 < 0 or (self.u4 == 0 and (self.u3 != 0 or self.u2 <= 0)):
            raise ValueError("Potential must be confining")
        if self.response not in {"exponential", "quadratic", "saturating"}:
            raise ValueError("Unknown response")

    @property
    def dc(self):
        return self.c_d * self.sigma_x * np.sqrt(self.tau_d)

    @property
    def noise(self):
        return self.sigma_y*np.sqrt(2*np.array(self.gamma)/np.array(self.a))

    def spectrum(self, n, m, breadth_y=10.0, breadth_x=10.0, preserve="integrated"):
        if n < 1 or m < 1 or min(breadth_y, breadth_x) < 1:
            raise ValueError("Positive mode counts and breadth >= 1 required")
        if preserve not in {"integrated", "zero_lag"}:
            raise ValueError("Unknown normalization")
        if any(self.cross_b):
            raise ValueError("Specify state-dependent amplitudes after changing the spectrum")
        def change(weights, rates, count, breadth):
            center = np.exp(np.log(rates).mean())
            r = center * (np.geomspace(1/np.sqrt(breadth), np.sqrt(breadth), count)
                          if count > 1 else np.ones(1))
            w = (r*np.sum(np.array(weights)/rates)/count if preserve == "integrated"
                 else np.full(count, np.sum(weights)/count))
            return tuple(w), tuple(r)
        a, gamma = change(self.a, np.array(self.gamma), n, breadth_y)
        c, lam = change(self.c, np.array(self.lam), m, breadth_x)
        return replace(self, a=a, gamma=gamma, c=c, lam=lam, cross_b=(0.0,)*m)

    def strength_summary(self):
        return {"YY_integrated": float(np.sum(np.array(self.a)/self.gamma)),
                "YY_zero_lag": sum(self.a),
                "YX_integrated": float(np.sum(np.array(self.c)/self.lam)),
                "YX_zero_lag": sum(self.c),
                "force_variance": self.sigma_y**2*sum(self.a)}


class Model:
    def __init__(self, parameters=None, variant=Variant.GLE):
        self.variant = Variant(variant)
        p = parameters or Parameters()
        self.p = p.spectrum(1, 1, 1, 1) if self.variant == Variant.SINGLE else p
        self.n, self.m = len(self.p.a), len(self.p.c)
        self.dim = 2+self.n+self.m
        self.h = slice(2, 2+self.n)
        self.g = slice(2+self.n, self.dim)

    def force(self, y):
        p = self.p
        return p.u2*y+p.u3*y*y+p.u4*y*y*y

    def force_derivative(self, y):
        return self.p.u2+2*self.p.u3*y+3*self.p.u4*y*y

    def cross(self, y):
        return np.array(self.p.c)*(1+np.array(self.p.cross_b)*np.tanh(y[..., None]/self.p.cross_scale))

    def pool(self, y):
        return 2*expit(-y/self.p.y_rho)

    def response(self, D):
        p = self.p
        x = np.abs(D)/p.dc
        if p.response == "exponential":
            val = np.where(x < 1e-4, x*x*(0.5-x/6+x*x/24-x**3/120), x+np.expm1(-x))
        elif p.response == "quadratic":
            val = 0.5*x*x
        else:
            val = np.tanh(x)
        return p.q_star*np.sign(D)*val+p.pi0*D

    def counterflow(self, D, y):
        if self.variant == Variant.KYLE:
            return np.zeros_like(D)
        rho = 1.0 if self.variant == Variant.FRESH else self.pool(np.sign(D)*y)
        return rho*self.response(D)

    def counterflow_derivative(self, D, y):
        p = self.p
        if self.variant == Variant.KYLE:
            return np.zeros_like(D)
        x = abs(D)/p.dc
        if p.response == "exponential":
            derivative = -np.expm1(-x)
        elif p.response == "quadratic":
            derivative = x
        else:
            derivative = 1-np.tanh(x)**2
        rho = 1.0 if self.variant == Variant.FRESH else self.pool(np.sign(D)*y)
        return rho*(p.q_star*derivative/p.dc+p.pi0)

    def drift(self, t, z, schedule):
        p = self.p
        D, y = z[..., 0], z[..., 1]
        F = -self.force(y)-z[..., self.h]@np.array(p.a)+z[..., self.g].sum(-1)
        out = np.zeros_like(z)
        q = schedule.rate(t)
        out[..., 0] = (q-self.counterflow(D, y))/p.L0
        out[..., 1] = F
        out[..., self.h] = F[..., None]-np.array(p.gamma)*z[..., self.h]
        out[..., self.g] = -np.array(p.lam)*z[..., self.g]+self.cross(y)*np.asarray(q)[..., None]
        return out

    def finite_start(self, paths, rng):
        z = np.zeros((paths, self.dim))
        z[:, self.h] = -rng.normal(size=(paths, self.n))*self.p.sigma_y/np.sqrt(self.p.a)
        return z
