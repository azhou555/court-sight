"""Pytest session setup.

torch and xgboost each bundle their own OpenMP runtime; on Apple Silicon,
whichever one runs a parallel op second segfaults regardless of import
order. Pinning to a single OpenMP thread avoids both libraries touching
their thread pools, which sidesteps the conflict. Must be set before either
is imported, so it lives here in conftest.py (collected before test modules).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
