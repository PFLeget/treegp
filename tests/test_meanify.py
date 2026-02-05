from __future__ import print_function
import warnings
import numpy as np
import copy
import treegp
import os
import fitsio
import glob

from treegp_test_helper import timer
from treegp_test_helper import get_correlation_length_matrix
from treegp_test_helper import make_2d_grf


def make_average(coord=None, gp=True):
    if coord is None:
        x = np.linspace(0, 2048, 10)
        x, y = np.meshgrid(x, x)
        x = x.reshape(len(x) ** 2)
        y = y.reshape(len(y) ** 2)
    else:
        x = coord[:, 0]
        y = coord[:, 1]

    average = 0.02 + 5e-8 * (x - 1024) ** 2 + 5e-8 * (y - 1024) ** 2
    params = copy.deepcopy(average)

    if gp:
        from scipy.spatial.distance import pdist, squareform

        dists = squareform(pdist(np.array([x, y]).T))
        cov = 0.03**2 * np.exp(-0.5 * dists**2 / 300.0**2)

        # avoids to print warning from numpy when generated uge gaussian random fields.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params += np.random.multivariate_normal([0] * len(x), cov)

    return np.array([x, y]).T, params


@timer
def test_meanify():
    np.random.seed(42)
    nfields = 300
    ndata = 500
    coords = []
    fields = []

    for n in range(nfields):
        x = np.random.uniform(0, 2048, size=ndata)
        y = np.random.uniform(0, 2048, size=ndata)
        coord = np.array([x, y]).T
        coord, field = make_average(coord=coord, gp=True)
        coords.append(coord)
        fields.append(field)

    meanify = treegp.meanify(bin_spacing=40.0, statistics="mean")
    for n in range(nfields):
        meanify.add_field(coords[n], fields[n])
    meanify.meanify()
    meanify.save_results(name_output=os.path.join("outputs", "mean_gp_stat_mean.fits"))
    coord, param = make_average(coord=meanify.coords0, gp=False)
    np.testing.assert_allclose(param, meanify.params0, atol=2e-1)


@timer
def test_gpinterp_meanify():
    optimizer = ["log-likelihood", "anisotropic"]
    npoints = [600, 2000]
    noise = 0.01
    sigma = 2.0
    size = 0.5
    g1 = 0.2
    g2 = 0.2
    ker = "AnisotropicRBF"

    # Generate 2D gaussian random fields.
    L = get_correlation_length_matrix(size, g1, g2)
    invL = np.linalg.inv(L)
    kernel = "%f**2*%s" % ((sigma, ker))
    kernel += "(invLam={0!r})".format(invL)
    kernel_skl = treegp.eval_kernel(kernel)

    for n, opt in enumerate(optimizer):
        x, y, y_err = make_2d_grf(kernel_skl, noise=noise, seed=42, npoints=npoints[n])
        # add mean function
        coords0, y0 = make_average(coord=x, gp=False)
        y += y0

        # Do gp interpolation without hyperparameters
        # fitting (truth is put initially).
        gp = treegp.GPInterpolation(
            kernel=kernel,
            optimizer=opt,
            normalize=True,
            nbins=21,
            min_sep=0.0,
            max_sep=3.0,
            p0=[0.5, 0, 0],
            average_fits=os.path.join("inputs", "mean_gp_stat_mean.fits"),
        )
        gp.initialize(x, y, y_err=y_err)
        gp.solve()
        # test if found hyperparameters are close the true hyperparameters.
        np.testing.assert_allclose(kernel_skl.theta, gp.kernel.theta, atol=5e-1)

        # Predict at same position as the simulated data.
        # Predictions are strictily equal to the input data
        # in the case of no noise. With noise you should expect
        # to have a pull distribution with mean function arround 0
        # with a std<1 (you use the same data to train and validate, and
        # the data are well sample compared to the input correlation
        # length).
        y_predict, y_cov = gp.predict(x, return_cov=True)
        y_std = np.sqrt(np.diag(y_cov))
        pull = y - y_predict
        pull /= np.sqrt(y_err**2 + y_std**2)
        mean_pull = np.mean(pull)
        std_pull = np.std(pull)

        # Test that mean of the pull is close to zeros and std of the pull bellow 1.
        np.testing.assert_allclose(
            0.0, mean_pull, atol=3.0 * (std_pull) / np.sqrt(npoints[n])
        )
        if std_pull > 1.0:
            raise ValueError(
                "std_pull is > 1. Current value std_pull = %f" % (std_pull)
            )

def test_MeanifyEquivalence():
    
    # 1. Generate Synthetic Data
    np.random.seed(42)
    N = 10000
    # Create random clusters of data
    coords = np.concatenate([
        np.random.normal(loc=500, scale=200, size=(N, 2)),
        np.random.normal(loc=1500, scale=200, size=(N, 2))
    ])
    # Function: Z = X + Y + Noise
    params = coords[:, 0] + coords[:, 1] + np.random.normal(0, 10, 2*N)
    
    # Define bounds explicitly so we can force alignment
    bounds = (0, 2000, 0, 2000) # u_min, u_max, v_min, v_max
    nominal_spacing = 50.0


    # --- 1. RUN LEGACY CODE ---
    legacy = treegp.meanify(bin_spacing=nominal_spacing, statistics="mean")
    legacy.add_field(coords, params)
    
    # Pass explicit bounds to legacy to control the grid
    legacy.meanify(lu_min=bounds[0], lu_max=bounds[1],
                    lv_min=bounds[2], lv_max=bounds[3])

    # --- 2. EXTRACT LEGACY GRID PARAMETERS ---
    # We must replicate the "linspace" spacing of the legacy code
    # Legacy effective bin count is (nominal_nbin - 1)
    # Legacy effective spacing is (Max - Min) / (nominal_nbin - 1)
        
    u_edges = legacy._xedge
    v_edges = legacy._yedge
    
    eff_spacing_u = u_edges[1] - u_edges[0]
    eff_spacing_v = v_edges[1] - v_edges[0]
        
    # Assert spacings are close enough to treat as uniform
    np.testing.assert_almost_equal(eff_spacing_u, eff_spacing_v, decimal=5)
    
    # --- 3. RUN STREAMING CODE ---
    # Initialize with the EFFECTIVE spacing derived from Legacy
    stream = treegp.MeanifyStream(bin_spacing=eff_spacing_u, bounds=bounds)
    
    # Simulate streaming by chunking data
    chunk_size = 5000
    for i in range(0, len(coords), chunk_size):
        c_chunk = coords[i:i+chunk_size]
        p_chunk = params[i:i+chunk_size]
        stream.add_field(c_chunk, p_chunk)
        
    stream.meanify()

    # --- 4. COMPARE RESULTS ---
    
    # A. Check Grid Alignment (Centers)
    # Flatten arrays for easy comparison
    leg_u0 = legacy._u0.T # Legacy uses Transpose internally in some spots, check .T usage
    # Actually legacy: average = average.T, and u0, v0 meshgrid.
    # Stream: u_grid, v_grid meshgrid (ij indexing).
    
    # Let's verify shapes first
    print(f"Legacy Shape: {legacy._average.shape}")
    print(f"Stream Shape: {stream._average.shape}")
    
    # The legacy class transposes the average at the end: self._average = average.T
    # `binned_statistic_2d` returns (nx, ny). Transpose makes it (ny, nx).
    # My Streaming class creates (nx, ny) grid.
    # We need to be careful with orientation.
    
    # Compare Mean Functions
    # We compare only finite values (bins that had data)
    mask_legacy = np.isfinite(legacy._average)
    mask_stream = np.isfinite(stream._average)
        
    # Ensure masks match (same bins filled)
    # Note: If dimensions are swapped, we might need stream._average.T
    if legacy._average.shape != stream._average.shape:
        # Legacy does a .T transform, streaming might need it to match
        stream_avg_aligned = stream._average.T
    else:
        stream_avg_aligned = stream._average

    # Check where both have data
    common_mask = mask_legacy & np.isfinite(stream_avg_aligned)
    
    # 1. Assert we filled roughly the same bins
    np.testing.assert_(np.any(common_mask), msg="No overlapping bins found!")
    
    # 2. Compare Values
    diff = legacy._average[common_mask] - stream_avg_aligned[common_mask]
    mae = np.mean(np.abs(diff))
    
    print(f"Max Difference: {np.max(np.abs(diff))}")
    print(f"Mean Abs Error: {mae}")

    # Tolerances: floating point sums vs iterative sums can have tiny diffs
    np.testing.assert_allclose(
        legacy._average[common_mask], 
        stream_avg_aligned[common_mask], 
        rtol=1e-7, atol=1e-10,
        err_msg="Mean values do not match!"
    )


if __name__ == "__main__":
    #test_meanify()
    #test_gpinterp_meanify()
    test_MeanifyEquivalence()
