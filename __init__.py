"""
gle -- a clean framework for the GLE-volatility falsification battery.

Four pieces:
    DataManager   (datamanager.py) : every data source, one loader each, all
                                     returning a uniform Dataset.
    numerics      (numerics.py)    : the canonical Bayesian kernel block plus
                                     every shared statistic, one copy.
    GLETestSuite  (tests.py)       : one method per test, source-agnostic.
    GLEReport     (utils.py)       : every plot and table, one class.

Typical use:

    from gle import DataManager, GLETestSuite, GLEReport

    dm = DataManager()
    ds = dm.load_wrds_taq("SPY", "2022-01-11", grid_spacing_ms=10.0,
                          variance_window_sec=2.0, max_minutes=5)

    suite = GLETestSuite(ds)
    suite.t1_kernel_estimation(prior_type="original", L=50)
    suite.t2_two_exponents()

    rep = GLEReport(outdir="figs")
    rep.plot_t2_histogram(suite.results["t2"])
"""

from .dataset import Dataset
from .datamanager import DataManager
from .model import GLEModel
from .tests import GLETestSuite
from .utils import GLEReport
from . import numerics

__all__ = ["Dataset", "DataManager", "GLETestSuite", "GLEReport", "numerics", "GLEModel"]
