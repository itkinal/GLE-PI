"""Piecewise-affine deterministic rates with exact volume and memory integrals."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    rate: float
    slope: float = 0.0

    def __post_init__(self):
        if not (0 <= self.start < self.end) or not np.isfinite(
                [self.start, self.end, self.rate, self.slope]).all():
            raise ValueError("Require finite segment with 0 <= start < end")


@dataclass(frozen=True)
class Schedule:
    segments: tuple[Segment, ...]
    duration: float
    name: str = "custom"

    def __post_init__(self):
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be positive")
        last = 0.0
        for s in self.segments:
            if s.start < last or s.end > self.duration:
                raise ValueError("Segments must be ordered, nonoverlapping, within duration")
            last = s.end

    def rate(self, t):
        t = np.asarray(t)
        out = np.zeros_like(t, dtype=float)
        for s in self.segments:
            out += np.where((t >= s.start) & (t < s.end),
                            s.rate + s.slope * (t - s.start), 0.0)
        return out

    def volume(self, t=None):
        t = self.duration if t is None else t
        out = np.zeros_like(np.asarray(t), dtype=float)
        for s in self.segments:
            x = np.clip(np.asarray(t) - s.start, 0, s.end - s.start)
            out += s.rate * x + 0.5 * s.slope * x**2
        return out

    @property
    def breaks(self):
        return sorted({0.0, self.duration, *(s.start for s in self.segments),
                       *(s.end for s in self.segments)})

    def zero(self):
        # Preserve boundaries for common-random-number comparisons.
        return Schedule(tuple(Segment(s.start, s.end, 0) for s in self.segments),
                        self.duration, "no-order")

    def memory(self, t, rates, amplitudes, g0=None):
        """Analytic convolution for constant c; final axis indexes modes."""
        t = np.asarray(t)[..., None]
        rates, amplitudes = np.asarray(rates), np.asarray(amplitudes)
        out = np.exp(-t * rates) * (np.zeros_like(rates) if g0 is None else g0)
        for s in self.segments:
            x = np.clip(t - s.start, 0, s.end - s.start)
            decay = np.exp(-rates * np.maximum(t - s.end, 0))
            a = -np.expm1(-rates * x) / rates
            # Series avoids cancellation in the convolution of a ramp.
            z = rates * x
            b = np.where(abs(z) < 1e-4,
                         x*x*(0.5-z/6+z*z/24-z*z*z/120), (x-a)/rates)
            out += amplitudes * decay * (s.rate*a + s.slope*b)
        return out

    @classmethod
    def flat(cls, Q, T):
        return cls((Segment(0, T, Q/T),), T, "flat")

    @classmethod
    def front(cls, Q, T):
        return cls((Segment(0, T, 2*Q/T, -2*Q/T**2),), T, "front")

    @classmethod
    def back(cls, Q, T):
        return cls((Segment(0, T, 0, 2*Q/T**2),), T, "back")

    @classmethod
    def pause(cls, Q, T, fraction=0.3):
        if not 0 < fraction < 1:
            raise ValueError("pause fraction must be in (0, 1)")
        a, b = (1-fraction)*T/2, (1+fraction)*T/2
        return cls((Segment(0, a, Q/((1-fraction)*T)),
                    Segment(b, T, Q/((1-fraction)*T))), T, "pause")

    @classmethod
    def pause_fixed_rate(cls, Q, active_rate, gap):
        if Q*active_rate <= 0 or gap < 0:
            raise ValueError("Q and rate must have the same nonzero sign; gap >= 0")
        active = Q/active_rate
        return cls((Segment(0, active/2, active_rate),
                    Segment(active/2+gap, active+gap, active_rate)),
                   active+gap, "pause-fixed-rate")

    @classmethod
    def round_trip(cls, Q, T, gap=0.0):
        return cls((Segment(0, T, Q/T), Segment(T+gap, 2*T+gap, -Q/T)),
                   2*T+gap, "round-trip")
