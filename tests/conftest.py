"""Pytest session setup.

Import xgboost before any test module loads torch. On macOS/conda, if torch's
OpenMP runtime (libomp) initialises before xgboost's, xgboost segfaults during
DMatrix construction in fit(). Loading xgboost first lets its runtime bind
cleanly, after which torch can coexist in the same process. conftest.py is
imported during collection, ahead of the test modules, so this ordering holds
for the whole session.
"""

import xgboost  # noqa: F401
