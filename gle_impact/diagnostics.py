"""Paper diagnostics; candidate bands are not claims of verified scaling."""
from dataclasses import dataclass, asdict
import numpy as np


def estimate(samples):
    x = np.asarray(samples)
    return {"mean": float(x.mean()),
            "se": float(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else None}


def local_exponents(V, I):
    V, I = np.asarray(V), np.asarray(I)
    if np.any(V <= 0) or np.any(np.diff(V) <= 0):
        raise ValueError("Sizes must be positive and strictly increasing")
    d = np.full(len(V), np.nan)
    valid = (I[:-2] > 0) & (I[2:] > 0) & (I[1:-1] > 0)
    middle = np.full(len(V)-2, np.nan)
    middle[valid] = np.log(I[2:][valid]/I[:-2][valid])/np.log(V[2:][valid]/V[:-2][valid])
    d[1:-1] = middle
    return d


def largest_band(V, I, kind="square_root", tolerance=0.1, min_width=1.0, slope_tolerance=0.0):
    """Exhaustive connected-interval search with positive slopes and concavity.

    Endpoints need centered derivatives, so outermost size points are excluded.
    slope_tolerance is an absolute tolerance in impact/volume units.
    """
    V, I = np.asarray(V), np.asarray(I)
    if kind not in {"square_root", "power_law"}:
        raise ValueError("Unknown band criterion")
    d = local_exponents(V, I)
    slopes = np.diff(I)/np.diff(V)
    best = None
    for i in range(1, len(V)-1):
        for j in range(i+1, len(V)-1):
            width = np.log10(V[j]/V[i])
            dd = d[i:j+1]
            if width < min_width or not np.isfinite(dd).all() or np.any(I[i:j+1] <= 0):
                continue
            mean = float(dd.mean())
            target = 0.5 if kind == "square_root" else mean
            if np.max(abs(dd-target)) > tolerance or np.any(slopes[i:j] <= 0):
                continue
            if (mean < 1 or kind == "square_root") and np.any(np.diff(slopes[i-1:j+1]) > slope_tolerance):
                continue
            if best is None or width > best["width_decades"]:
                best = {"V_minus": float(V[i]), "V_plus": float(V[j]),
                        "width_decades": float(width), "mean_exponent": mean}
    return best


def response_summary(order, control, schedule):
    if not np.array_equal(order.times, control.times):
        raise ValueError("Responses must share observation times")
    sign = float(np.sign(schedule.volume()))
    if sign == 0:
        raise ValueError("Use round_trip_identity for a zero-net-volume schedule")
    Jpaths = order.states[:, :, 0]-control.states[:, :, 0]
    J = Jpaths.mean(axis=1)
    idx = int(np.argmin(abs(order.times-schedule.duration)))
    if abs(order.times[idx]-schedule.duration) > 1e-10:
        raise ValueError("Execution terminal time must be recorded")
    # For the one-sided schedules, integral |q| sgn(Q) J / |Q| = integral q J / |Q|.
    cost = (order.trading_cost[idx]-control.trading_cost[idx])/abs(float(schedule.volume()))
    out = {"terminal": estimate(sign*Jpaths[idx]),
           "peak_mean_response": float(np.max(sign*J[:idx+1])),
           "mean_pathwise_peak": float(np.max(sign*Jpaths[:idx+1], axis=0).mean()),
           "execution_displacement": estimate(cost),
           "counterflow_volume": estimate(order.cf_volume[idx]-control.cf_volume[idx]),
           "net_reversal_fraction": order.net_reversal_fraction}
    return out


def relaxation(times, J, T, levels=(0.5, 0.1), tolerance=1e-10):
    idx = np.flatnonzero(np.isclose(times, T))
    if len(idx) != 1 or abs(J[idx[0]]) <= tolerance:
        return {"resolved": False, "reason": "terminal response unresolved"}
    i = idx[0]
    tau, R = times[i:]-T, J[i:]/J[i]
    crossings = {}
    for alpha in levels:
        hits = np.flatnonzero(R <= alpha)
        if not len(hits):
            crossings[str(alpha)] = None
        else:
            k = int(hits[0])
            crossings[str(alpha)] = float(tau[k] if k == 0 else
                tau[k-1]+(alpha-R[k-1])*(tau[k]-tau[k-1])/(R[k]-R[k-1]))
    return {"resolved": True, "crossings": crossings, "residual_ratio": float(R[-1]),
            "observation_extension": float(tau[-1])}


def round_trip_identity(result, L0):
    left = result.trading_cost[-1]
    right = L0/2*(result.states[-1, :, 0]**2-result.states[0, :, 0]**2)+result.cf_cost[-1]
    return {"cost": estimate(left), "rhs": estimate(right),
            "max_pathwise_identity_error": float(np.max(abs(left-right)))}
