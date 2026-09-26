"""PyTorch backward-equation PINN, including deterministic-g reduction.

Full state: (D,Y,h_1,...,h_N,g_1,...,g_M).
Reduced state: (D,Y,h_1,...,h_N), only for constant c and deterministic g0.
No spatial boundary condition is invented for the unbounded-state problem.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .model import Model, Variant
from .schedules import Schedule

DTYPE = torch.float64


class TorchCoefficients:
    def __init__(self, model: Model, schedule: Schedule, reduced=True, g0=None, device="cpu"):
        self.model, self.p, self.schedule = model, model.p, schedule
        if reduced and any(model.p.cross_b):
            raise ValueError("State-dependent cross-memory requires full state")
        self.reduced, self.device = reduced, torch.device(device)
        self.dim = 2+model.n if reduced else model.dim
        self.a, self.gamma, self.c, self.lam, self.b, self.sigma = (
            self.tensor(x) for x in (self.p.a, self.p.gamma, self.p.c, self.p.lam,
                                    self.p.cross_b, self.p.noise))
        self.g0 = self.tensor(np.zeros(model.m) if g0 is None else g0)
        if self.g0.shape != (model.m,):
            raise ValueError("g0 must be one deterministic mode vector")

    def tensor(self, x):
        return torch.as_tensor(x, dtype=DTYPE, device=self.device)

    def rate(self, t):
        q = torch.zeros_like(t)
        for s in self.schedule.segments:
            q = q+torch.where((t >= s.start) & (t < s.end), s.rate+s.slope*(t-s.start), 0.0)
        return q

    def memory(self, t):
        out = torch.exp(-t*self.lam)*self.g0
        for s in self.schedule.segments:
            x = torch.clamp(t-s.start, 0, s.end-s.start)
            decay = torch.exp(-self.lam*torch.clamp(t-s.end, min=0))
            z = self.lam*x
            a = -torch.expm1(-z)/self.lam
            b = torch.where(abs(z) < 1e-4, x*x*(0.5-z/6+z*z/24-z**3/120), (x-a)/self.lam)
            out = out+self.c*decay*(s.rate*a+s.slope*b)
        return out

    def full(self, t, z):
        return torch.cat((z, self.memory(t)), dim=1) if self.reduced else z

    def pool(self, y):
        return 2*torch.sigmoid(-y/self.p.y_rho)

    def counterflow(self, z):
        D, y, p = z[:, :1], z[:, 1:2], self.p
        if self.model.variant == Variant.KYLE:
            return D*0
        x = abs(D)/p.dc
        if p.response == "exponential":
            a = torch.where(x < 1e-4, x*x*(0.5-x/6+x*x/24-x**3/120), x+torch.expm1(-x))
        elif p.response == "quadratic":
            a = 0.5*x*x
        else:
            a = torch.tanh(x)
        response = p.q_star*torch.sign(D)*a+p.pi0*D
        rho = 1 if self.model.variant == Variant.FRESH else self.pool(torch.sign(D)*y)
        return rho*response

    def drift(self, t, z):
        full, p, m = self.full(t, z), self.p, self.model
        y, h, g = full[:, 1:2], full[:, m.h], full[:, m.g]
        F = -(p.u2*y+p.u3*y*y+p.u4*y**3)-(h*self.a).sum(1, keepdim=True)+g.sum(1, keepdim=True)
        out = [(self.rate(t)-self.counterflow(full))/p.L0, F, F-self.gamma*h]
        if not self.reduced:
            out.append(-self.lam*g+self.c*(1+self.b*torch.tanh(y/p.cross_scale))*self.rate(t))
        return torch.cat(out, dim=1)


class Observables:
    """One network output per terminal/running functional; custom subclasses allowed."""
    def __init__(self, names=("D", "Y", "pool", "cf_volume"), order_sign=1):
        self.names, self.order_sign = tuple(names), order_sign
        if order_sign not in (-1, 1):
            raise ValueError("order_sign must be +/-1, also for the no-order control")

    def terminal(self, coef, t, z):
        f = coef.full(t, z)
        zero = f[:, :1]*0
        values = []
        for name in self.names:
            if name == "D": v = f[:, :1]
            elif name == "D2": v = f[:, :1]**2
            elif name == "Y": v = f[:, 1:2]
            elif name == "Y2": v = f[:, 1:2]**2
            elif name == "pool": v = coef.pool(self.order_sign*f[:, 1:2])
            elif name.startswith("h_"): v = f[:, 2+int(name[2:]):3+int(name[2:])]
            elif name.startswith("g_"): v = f[:, 2+coef.model.n+int(name[2:]):3+coef.model.n+int(name[2:])]
            elif name in {"cf_volume", "cost", "cf_cost"}: v = zero
            else: raise ValueError(f"Unknown observable: {name}")
            if v.shape[1] != 1: raise ValueError(f"Invalid observable index: {name}")
            values.append(v)
        return torch.cat(values, dim=1)

    def running(self, coef, t, z):
        cf, D = coef.counterflow(z), z[:, :1]
        return torch.cat([cf if name == "cf_volume" else coef.rate(t)*D if name == "cost"
                          else cf*D if name == "cf_cost" else D*0 for name in self.names], dim=1)


@dataclass(frozen=True)
class PINNConfig:
    width: int = 64
    depth: int = 3
    epochs: int = 2000
    batch_size: int = 512
    learning_rate: float = 1e-3
    seed: int = 1729
    validate_every: int = 100
    device: str = "cpu"
    lbfgs_steps: int = 0


@dataclass
class Domain:
    lower: np.ndarray
    upper: np.ndarray
    cloud: np.ndarray | None = None  # [time, state] in physical coordinates

    def __post_init__(self):
        self.lower, self.upper = np.asarray(self.lower), np.asarray(self.upper)
        if self.lower.shape != self.upper.shape or np.any(self.upper <= self.lower):
            raise ValueError("Require strictly ordered domain bounds")

    @classmethod
    def from_results(cls, *results, reduced_dim=None, expansion=1.5):
        """Union of order/control pilot samples; box remains an explicit finite approximation."""
        rows = []
        for r in results:
            states = r.states if reduced_dim is None else r.states[:, :, :reduced_dim]
            t = np.broadcast_to(r.times[:, None, None], (*states.shape[:2], 1))
            rows.append(np.concatenate((t, states), axis=2).reshape(-1, states.shape[-1]+1))
        cloud = np.concatenate(rows)
        low, high = cloud[:, 1:].min(0), cloud[:, 1:].max(0)
        mid, half = (low+high)/2, np.maximum((high-low)*expansion/2, 0.1)
        return cls(mid-half, mid+half, cloud)

    def sample(self, n, start, end, rng):
        z = rng.uniform(self.lower, self.upper, size=(n, len(self.lower)))
        t = rng.uniform(start, end, size=(n, 1))
        if self.cloud is not None:
            rows = self.cloud[(self.cloud[:, 0] >= start) & (self.cloud[:, 0] <= end)]
            if len(rows):
                k = n//2
                chosen = rows[rng.integers(len(rows), size=k)].copy()
                chosen[:, 0] = np.clip(chosen[:, 0], start+1e-7*(end-start), end-1e-7*(end-start))
                t[:k], z[:k] = chosen[:, :1], chosen[:, 1:]
        return np.column_stack((t, z))


class TimeBlock(nn.Module):
    def __init__(self, coef, observable, domain, start, end, config, output_scales, continuation=None):
        super().__init__()
        self.coef, self.observable = coef, observable
        self.start, self.end = start, end
        # Register continuation for checkpointing; frozen parameters still permit state derivatives.
        self.continuation = continuation
        if continuation is not None:
            for param in continuation.parameters(): param.requires_grad_(False)
        self.register_buffer("center", coef.tensor((domain.lower+domain.upper)/2))
        self.register_buffer("scale", coef.tensor((domain.upper-domain.lower)/2))
        self.register_buffer("output_scales", coef.tensor(output_scales))
        layers = [nn.Linear(coef.dim+1, config.width), nn.Tanh()]
        for _ in range(config.depth-1): layers += [nn.Linear(config.width, config.width), nn.Tanh()]
        layers += [nn.Linear(config.width, len(observable.names))]
        self.net = nn.Sequential(*layers).to(device=coef.device, dtype=DTYPE)

    def forward(self, tz):
        t, z = tz[:, :1], tz[:, 1:]
        at_end = torch.cat((torch.full_like(t, self.end), z), dim=1)
        terminal = (self.observable.terminal(self.coef, at_end[:, :1], z) if self.continuation is None
                    else self.continuation(at_end))
        x = torch.cat((2*(t-self.start)/(self.end-self.start)-1, (z-self.center)/self.scale), dim=1)
        return terminal+(self.end-t)/(self.end-self.start)*self.output_scales*self.net(x)


def gradient(value, inputs):
    if not value.requires_grad:
        return torch.zeros_like(inputs)
    result = torch.autograd.grad(value.sum(), inputs, create_graph=True,
                                 retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(inputs) if result is None else result


def generator_value(value, tz, coef):
    """Compute (partial_t + L) value, exact diagonal Hessian in h only."""
    first = gradient(value, tz)
    result = first[:, :1]+(coef.drift(tz[:, :1], tz[:, 1:])*first[:, 1:]).sum(1, keepdim=True)
    for i in range(coef.model.n):
        column = 3+i  # time, D, Y, then h_i
        if coef.p.noise[i] != 0:
            second = gradient(first[:, column:column+1], tz)[:, column:column+1]
            result = result+0.5*coef.sigma[i]**2*second
    return result


class BackwardPINN:
    def __init__(self, coefficients, observables, domain, horizon, config=None, output_scales=None):
        self.coef, self.observables, self.domain = coefficients, observables, domain
        self.horizon, self.config = float(horizon), config or PINNConfig()
        if self.horizon <= 0 or len(domain.lower) != coefficients.dim:
            raise ValueError("Invalid horizon or domain dimension")
        self.output_scales = np.ones(len(observables.names)) if output_scales is None else np.asarray(output_scales)
        if self.output_scales.shape != (len(observables.names),) or np.any(self.output_scales <= 0):
            raise ValueError("One positive output scale per observable required")
        self.blocks, self.history = [], []

    def residual(self, block, points):
        tz = points.detach().clone().requires_grad_(True)
        u = block(tz)
        r = self.observables.running(self.coef, tz[:, :1], tz[:, 1:])
        return torch.cat([generator_value(u[:, j:j+1], tz, self.coef)+r[:, j:j+1]
                          for j in range(u.shape[1])], dim=1)

    def fit(self):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng, validation_rng = np.random.default_rng(cfg.seed), np.random.default_rng(cfg.seed+1)
        breaks = sorted({0.0, self.horizon, *(b for b in self.coef.schedule.breaks if 0 < b < self.horizon)})
        continuation = None
        started = time.perf_counter()
        self.blocks, self.history = [], []
        for start, end in reversed(list(zip(breaks[:-1], breaks[1:]))):
            block = TimeBlock(self.coef, self.observables, self.domain, start, end,
                              cfg, self.output_scales, continuation)
            optimizer = torch.optim.Adam(block.net.parameters(), lr=cfg.learning_rate)
            validation = self.coef.tensor(self.domain.sample(cfg.batch_size, start, end, validation_rng))
            residual_scale = self.coef.tensor(self.output_scales/(end-start))
            for epoch in range(cfg.epochs):
                points = self.coef.tensor(self.domain.sample(cfg.batch_size, start, end, rng))
                optimizer.zero_grad(set_to_none=True)
                residual = self.residual(block, points)/residual_scale
                loss = (residual**2).mean()
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite PINN loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(block.net.parameters(), 10.0)
                optimizer.step()
                if epoch % cfg.validate_every == 0 or epoch == cfg.epochs-1:
                    val = self.residual(block, validation).detach()
                    self.history.append({"start": start, "end": end, "epoch": epoch,
                                         "train_scaled_mse": float(loss.detach()),
                                         "validation_physical_rmse": val.square().mean(0).sqrt().cpu().tolist()})
            if cfg.lbfgs_steps > 0:
                # L-BFGS needs a fixed objective during its line search.
                polish_points = self.coef.tensor(self.domain.sample(4*cfg.batch_size, start, end, rng))
                polish = torch.optim.LBFGS(block.net.parameters(), max_iter=cfg.lbfgs_steps,
                                          line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                          tolerance_change=1e-12, history_size=50)
                def closure():
                    polish.zero_grad(set_to_none=True)
                    value = (self.residual(block, polish_points)/residual_scale).square().mean()
                    if not torch.isfinite(value): raise FloatingPointError("Nonfinite L-BFGS loss")
                    value.backward()
                    return value
                polish.step(closure)
                val = self.residual(block, validation).detach()
                self.history.append({"start": start, "end": end, "stage": "lbfgs",
                                     "validation_physical_rmse": val.square().mean(0).sqrt().cpu().tolist()})
            self.blocks.insert(0, block)
            continuation = block
        self.training_seconds = time.perf_counter()-started
        return self

    def predict(self, t, states):
        if not self.blocks: raise RuntimeError("Call fit first")
        if not 0 <= t <= self.horizon: raise ValueError("Time outside trained horizon")
        block = next(b for b in self.blocks if b.start <= t <= b.end)
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != self.coef.dim:
            raise ValueError("State array has wrong dimension")
        tz = self.coef.tensor(np.column_stack((np.full(len(states), t), states)))
        with torch.no_grad(): return block(tz).cpu().numpy()

    def average(self, initial):
        values = self.predict(0.0, initial)
        return {"mean": values.mean(0),
                "initial_quadrature_se": values.std(0, ddof=1)/np.sqrt(len(values))}

    def save(self, path):
        """Save metadata and weights; reconstruct same problem before load_state_dict."""
        torch.save({"parameters": asdict(self.coef.p), "variant": str(self.coef.model.variant),
                    "schedule": asdict(self.coef.schedule), "horizon": self.horizon,
                    "reduced": self.coef.reduced, "g0": self.coef.g0,
                    "observables": self.observables.names, "order_sign": self.observables.order_sign,
                    "config": asdict(self.config), "lower": self.domain.lower.tolist(),
                    "upper": self.domain.upper.tolist(), "output_scales": self.output_scales.tolist(),
                    "blocks": [b.state_dict() for b in self.blocks], "history": self.history,
                    "training_seconds": self.training_seconds}, Path(path))
