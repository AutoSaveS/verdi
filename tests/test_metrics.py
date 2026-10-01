"""Checks of the evaluation helpers against Appendix C.2."""
import numpy as np

from verdi.eval.metrics import cross_modal_pair_checks, expected_calibration_error


def test_cross_modal_pairs_use_the_appendix_thresholds():
    # Pair values reported in Section 5.2: all linked pairs above, m3-m6 below.
    r2 = {("m2", "m6"): 0.42, ("m5", "m2"): 0.38, ("m1", "m2"): 0.36,
          ("m1", "m5"): 0.31, ("m3", "m6"): 0.05}
    assert all(cross_modal_pair_checks(r2).values())
    r2[("m2", "m6")] = 0.34                      # below its 0.35 threshold
    assert not cross_modal_pair_checks(r2)["m2-m6"]


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
    rng = np.random.default_rng(0)
    correct = rng.random(p.size) < p
    assert expected_calibration_error(p, correct) < 0.02
