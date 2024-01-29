from sps.gp import GP

# from sps.metrics import maximum_mean_discrepancy
from sps.utils import build_grid


def test_gp_simulator():
    gp, grid, batch_size = GP(), build_grid(), 16
    samples = gp.simulate(grid, batch_size)
    samples_approx = gp.simulate(grid, batch_size, approx=True)
    assert samples.shape == samples_approx.shape
    # assert maximum_mean_discrepancy(samples, samples_approx) < 0.1, "MMD too high!"
