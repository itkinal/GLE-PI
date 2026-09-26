"""Vectorized independent-path simulation with a drift-implicit midpoint step.

Additive noise acts only on h. A scalar nonlinear solve advances Y and the
memory states jointly; a second scalar solve advances D. No drift is clipped.
The scheme is a numerical approximation, not an exact SDE transition.
"""
from dataclasses import dataclass
import numpy as np
from .model import Model
from .schedules import Schedule


@dataclass(frozen=True)
class SimulationConfig:
    paths: int = 512
    dt: float = 0.02
    observations: int = 201
    seed: int = 1729
    root_tolerance: float = 1e-11
    root_iterations: int = 60

    def __post_init__(self):
        if self.paths < 1 or self.observations < 2 or self.dt <= 0:
            raise ValueError("Invalid simulation configuration")


@dataclass
class SimulationResult:
    times: np.ndarray
    states: np.ndarray  # [observation, independent path, full state]
    cf_volume: np.ndarray
    trading_cost: np.ndarray
    cf_cost: np.ndarray
    min_pool: np.ndarray
    max_pool: np.ndarray
    net_reversal_fraction: float

    @property
    def mean(self):
        return self.states.mean(axis=1)

    @property
    def se(self):
        if self.states.shape[1] == 1:
            return np.full_like(self.mean, np.nan)
        return self.states.std(axis=1, ddof=1)/np.sqrt(self.states.shape[1])


class MonteCarlo:
    def __init__(self, model: Model, config=None):
        self.model, self.config = model, config or SimulationConfig()

    def step(self, z, t, dt, schedule, normals):
        model, cfg, p = self.model, self.config, self.model.p
        D, y, h, g = z[:, 0], z[:, 1], z[:, model.h], z[:, model.g]
        q = float(schedule.rate(t+dt/2))
        den_h = 1+np.array(p.gamma)*dt/2
        den_g = 1+np.array(p.lam)*dt/2
        noise = p.noise*np.sqrt(dt)*normals
        hm0 = (h-noise/2)/den_h
        gm0 = g/den_g
        x = y.copy()  # midpoint Y
        # Implicit equation: 2(x-y)/dt = -U'(x)-a.hmid+sum(gmid).
        for _ in range(cfg.root_iterations):
            hm = hm0+(x-y)[:, None]/den_h
            gm = gm0+dt*q*model.cross(x)/(2*den_g)
            F = -model.force(x)-hm@np.array(p.a)+gm.sum(-1)
            residual = 2*(x-y)/dt-F
            dc = (np.array(p.c)*np.array(p.cross_b)/p.cross_scale
                  * (1-np.tanh(x[:, None]/p.cross_scale)**2))
            jac = 2/dt+model.force_derivative(x)+np.sum(np.array(p.a)/den_h)-np.sum(dt*q*dc/(2*den_g), axis=-1)
            if np.any(jac <= 0):
                raise RuntimeError("Latent midpoint equation not monotone; reduce dt for this parameter/order")
            dx = residual/jac
            x -= dx
            if np.max(abs(dx)/(1+abs(x))) < cfg.root_tolerance:
                break
        else:
            raise RuntimeError("Latent midpoint solve failed; reduce dt")
        hm = hm0+(x-y)[:, None]/den_h
        gm = gm0+dt*q*model.cross(x)/(2*den_g)
        # Dmid + dt/(2 L0) cf(Dmid,Ymid) = Dold + dt*q/(2 L0).
        rhs = D+dt*q/(2*p.L0)
        lo, hi = np.minimum(rhs, 0), np.maximum(rhs, 0)
        dmid = rhs.copy()
        for _ in range(cfg.root_iterations):
            cf = model.counterflow(dmid, x)
            f = dmid+dt*cf/(2*p.L0)-rhs
            if np.max(abs(f)/(1+abs(rhs))) < cfg.root_tolerance:
                break
            lo = np.where(f < 0, dmid, lo)
            hi = np.where(f >= 0, dmid, hi)
            proposal = dmid-f/(1+dt*model.counterflow_derivative(dmid, x)/(2*p.L0))
            dmid = np.where((proposal >= lo) & (proposal <= hi), proposal, (lo+hi)/2)
        else:
            raise RuntimeError("Displacement midpoint solve failed")
        new = np.empty_like(z)
        new[:, 0], new[:, 1] = 2*dmid-D, 2*x-y
        new[:, model.h], new[:, model.g] = 2*hm-h, 2*gm-g
        if not np.isfinite(new).all():
            raise FloatingPointError("Nonfinite state; reduce dt and inspect parameters")
        return new, cf*dt, q*dmid*dt, cf*dmid*dt, model.pool(np.sign(dmid)*x), (q-cf)*q < 0

    def run(self, schedule, horizon=None, initial=None, times=None, cost_schedule=None):
        cfg, model = self.config, self.model
        H = schedule.duration if horizon is None else float(horizon)
        if H <= 0:
            raise ValueError("Positive horizon required")
        rng = np.random.default_rng(cfg.seed)
        # Draw initial randomness even with supplied states, keeping future noise aligned.
        default = model.finite_start(cfg.paths, rng)
        z = default if initial is None else np.asarray(initial, dtype=float).copy()
        if z.shape != (cfg.paths, model.dim) or not np.isfinite(z).all():
            raise ValueError("Initial ensemble must have shape (paths, full state dimension)")
        requested = np.linspace(0, H, cfg.observations) if times is None else np.asarray(times)
        if np.any(requested < 0) or np.any(requested > H):
            raise ValueError("Observation times outside horizon")
        # Every switch is both an integration boundary and an observation.
        observations = np.unique(np.r_[0, requested, H, [b for b in schedule.breaks if b <= H]])
        states, cvs, costs, ccosts = [z.copy()], [np.zeros(cfg.paths)], [np.zeros(cfg.paths)], [np.zeros(cfg.paths)]
        cv, cost, ccost = (np.zeros(cfg.paths) for _ in range(3))
        low, high = np.full(cfg.paths, np.inf), np.full(cfg.paths, -np.inf)
        reversals = active_steps = 0
        for a, b in zip(observations[:-1], observations[1:]):
            n = max(1, int(np.ceil((b-a)/cfg.dt)))
            dt = (b-a)/n
            for k in range(n):
                t = a+k*dt
                old_D = z[:, 0].copy()
                z, dv, dc, dr, pool, reverse = self.step(z, t, dt, schedule, rng.normal(size=(cfg.paths, model.n)))
                if cost_schedule is not None:
                    dc = float(cost_schedule.rate(t+dt/2))*(old_D+z[:, 0])*dt/2
                cv += dv
                cost += dc
                ccost += dr
                low, high = np.minimum(low, pool), np.maximum(high, pool)
                if schedule.rate(t+dt/2) != 0:
                    reversals += int(reverse.sum())
                    active_steps += cfg.paths
            states.append(z.copy()); cvs.append(cv.copy())
            costs.append(cost.copy()); ccosts.append(ccost.copy())
        return SimulationResult(observations, np.array(states), np.array(cvs),
                                np.array(costs), np.array(ccosts), low, high,
                                reversals/max(1, active_steps))

    def paired(self, schedule, horizon=None, initial=None):
        order = self.run(schedule, horizon, initial)
        control = self.run(schedule.zero(), horizon, initial, times=order.times, cost_schedule=schedule)
        return order, control

    def burn_in(self, duration=100.0):
        """Joint no-order ensemble. Caller must check duration/dt sensitivity."""
        return self.run(Schedule.flat(0.0, duration), times=np.array([0, duration])).states[-1]
