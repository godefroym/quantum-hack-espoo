"""Shared, self-documenting building blocks for the end-to-end demonstration scripts.

The two entry-point scripts (``scripts/run_mvp.py`` and ``scripts/run_demonstration.py``)
stay thin; the load-bearing logic lives here so it is written once and tested by running it.

Layout:

* :mod:`._bootstrap`  -- put ``src/`` on the path and pin matplotlib/cache dirs under ``outputs/``.
* :mod:`._specs`      -- choose the demonstration specs and the achievable (Fréchet) corr ceiling.
* :mod:`._second_order` -- criterion 1: marginal + correlation match vs the achievable ceiling.
* :mod:`._higher_order`  -- criteria 2 & 3: higher-order/tail discriminators and cascade-tail movement.
* :mod:`._scale`        -- the n = 54 homogeneous mean-field oracle validation (the scale story).
* :mod:`._report`       -- formatting tables, verdicts, and writing the ``outputs/`` artifacts.
"""
