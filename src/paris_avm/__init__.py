"""Automated Valuation Model for Paris residential listings.

Pipeline: `data` -> `cleaning` -> `features` -> `modeling` -> `evaluation`,
with `plots` for the figures and `config` holding every threshold.
"""

from . import cleaning, config, data, evaluation, features, modeling, plots  # noqa: F401

__version__ = "1.0.0"
__all__ = ["config", "data", "cleaning", "features", "modeling", "evaluation", "plots"]
