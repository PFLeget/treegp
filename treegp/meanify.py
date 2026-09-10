"""
.. module:: meanify
"""

import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.stats import biweight_location
import fitsio
import copy


def biweight(x, c=6.0):
    """Biweight location estimate of a 1d sample (Beers, Flynn & Gebhardt
    1990, AJ 100, 32). One-step Tukey biweight centered on the median with
    a MAD-based scale; points beyond ``c * MAD`` get zero weight.

    Returns nan on an empty sample, as required by ``binned_statistic_2d``
    for bins without data.

    :param x: 1d array-like sample.
    :param c: Tuning constant. (default: 6.)
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(biweight_location(x, c=c))


def median_clipped(x, nsigma=3.0, maxiters=5):
    """Median of a 1d sample with iterative MAD clipping.

    At each iteration the median and the normalized MAD
    (``1.4826 * MAD``, i.e. sigma for a Gaussian) of the surviving points
    are computed, and points farther than ``nsigma`` sigma from the median
    are rejected. Iterations stop when no more points are rejected, when
    the MAD is zero, or after ``maxiters`` iterations.

    Returns nan on an empty sample, as required by ``binned_statistic_2d``
    for bins without data.

    :param x:        1d array-like sample.
    :param nsigma:   Clipping threshold in units of normalized MAD. (default: 3.)
    :param maxiters: Maximum number of clipping iterations. (default: 5)
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    mask = np.ones(x.size, dtype=bool)
    for _ in range(maxiters):
        med = np.median(x[mask])
        mad = np.median(np.abs(x[mask] - med))
        if mad == 0:
            break
        new_mask = mask & (np.abs(x - med) <= nsigma * 1.4826 * mad)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    return float(np.median(x[mask]))


class meanify(object):
    """Take data, build a spatial average, and write output average.

    :param bin_spacing: Bin_size, resolution on the mean function. (default=120.)
    :param statistics:  Statistics used to compute the mean. (default=mean)
                        Supported: "mean", "median", "weighted", "biweight",
                        "median_clipped".
                        "biweight" is the outlier-resistant biweight location
                        of Beers, Flynn & Gebhardt 1990 (see ``biweight``).
                        "median_clipped" is a median with iterative MAD
                        clipping (see ``median_clipped``). For both, wrms is
                        set to zero, as for "mean" and "median".
    :param bounds:      Optional tuple (u_min, u_max, v_min, v_max).
                        When provided with statistics="mean", enables O(1) memory
                        streaming mode using accumulators instead of storing all data.
    :param biweight_c:    Tuning constant of the biweight. (default: 6.)
    :param clip_nsigma:   Clipping threshold, in normalized MAD units, used by
                          "median_clipped". (default: 3.)
    :param clip_maxiters: Maximum number of clipping iterations used by
                          "median_clipped". (default: 5)
    """

    SUPPORTED_STATISTICS = ["mean", "median", "weighted", "biweight", "median_clipped"]

    def __init__(
        self,
        bin_spacing=120.0,
        statistics="mean",
        bounds=None,
        biweight_c=6.0,
        clip_nsigma=3.0,
        clip_maxiters=5,
    ):
        self.bin_spacing = bin_spacing

        if statistics not in self.SUPPORTED_STATISTICS:
            raise ValueError(
                "%s is not a supported statistic (supported: %s)"
                % (statistics, ", ".join(self.SUPPORTED_STATISTICS))
            )
        self.stat_used = statistics
        self.biweight_c = biweight_c
        self.clip_nsigma = clip_nsigma
        self.clip_maxiters = clip_maxiters

        # Determine if we can use streaming mode
        self._use_streaming = (statistics == "mean") and (bounds is not None)

        if self._use_streaming:
            self._init_streaming(bounds)
        else:
            self._init_legacy()

    def _init_streaming(self, bounds):
        """Initialize streaming accumulators for O(1) memory usage."""
        self.bin_spacing = float(self.bin_spacing)
        self.lu_min, self.lu_max, self.lv_min, self.lv_max = bounds

        # Replicate legacy grid logic exactly
        n_edges_u = int((self.lu_max - self.lu_min) / self.bin_spacing)
        n_edges_v = int((self.lv_max - self.lv_min) / self.bin_spacing)

        # Generate edges exactly like legacy
        self._xedge = np.linspace(self.lu_min, self.lu_max, n_edges_u)
        self._yedge = np.linspace(self.lv_min, self.lv_max, n_edges_v)

        # Actual number of bins is edges - 1
        self.nbin_u = len(self._xedge) - 1
        self.nbin_v = len(self._yedge) - 1

        # Calculate effective spacing for arithmetic binning
        self.dx = self._xedge[1] - self._xedge[0]
        self.dy = self._yedge[1] - self._yedge[0]

        # Accumulators (u, v) -> (x, y) order
        shape = (self.nbin_u, self.nbin_v)
        self._grid_sum = np.zeros(shape, dtype=np.float64)
        self._grid_sum_sq = np.zeros(shape, dtype=np.float64)
        self._grid_count = np.zeros(shape, dtype=np.int64)

    def _init_legacy(self):
        """Initialize legacy mode with data lists."""
        self.coords = []
        self.params = []
        self.params_err = []

    def add_field(self, coord, param, params_err=None):
        """
        Add new data to compute the mean function.

        :param coord: Array of coordinate of the parameter.
        :param param: Array of parameter.
        :param params_err: Array of parameter errors (required for weighted statistics).
        """
        if np.shape(coord)[1] != 2:
            raise ValueError("meanify is supported only in 2d for the moment.")

        if self._use_streaming:
            self._add_field_streaming(coord, param)
        else:
            self._add_field_legacy(coord, param, params_err)

    def _add_field_streaming(self, coord, param):
        """Add data using streaming accumulators."""
        # Arithmetic binning using effective spacing
        u_idx = ((coord[:, 0] - self.lu_min) / self.dx).astype(int)
        v_idx = ((coord[:, 1] - self.lv_min) / self.dy).astype(int)

        # Handle inclusive right edge (match binned_statistic behavior)
        on_u_edge = coord[:, 0] == self.lu_max
        on_v_edge = coord[:, 1] == self.lv_max
        u_idx[on_u_edge] = self.nbin_u - 1
        v_idx[on_v_edge] = self.nbin_v - 1

        # Filter valid points
        valid_mask = (
            (u_idx >= 0)
            & (u_idx < self.nbin_u)
            & (v_idx >= 0)
            & (v_idx < self.nbin_v)
            & np.isfinite(param)
        )

        if not np.any(valid_mask):
            return

        u_idx = u_idx[valid_mask]
        v_idx = v_idx[valid_mask]
        p = param[valid_mask]

        # Accumulate
        np.add.at(self._grid_sum, (u_idx, v_idx), p)
        np.add.at(self._grid_sum_sq, (u_idx, v_idx), p**2)
        np.add.at(self._grid_count, (u_idx, v_idx), 1)

    def _add_field_legacy(self, coord, param, params_err):
        """Add data to lists for legacy processing."""
        self.coords.append(coord[np.isfinite(param)])
        self.params.append(param[np.isfinite(param)])
        if self.stat_used == "weighted":
            if params_err is None:
                raise ValueError("Need an associated error to params")
            else:
                self.params_err.append(params_err[np.isfinite(param)])

    def meanify(self, lu_min=None, lu_max=None, lv_min=None, lv_max=None):
        """
        Compute the mean function.

        :param lu_min, lu_max, lv_min, lv_max: Bounds for the grid.
            In streaming mode, these are ignored (bounds set at init).
            In legacy mode, if not provided, computed from data.
        """
        if self._use_streaming:
            self._meanify_streaming()
        else:
            self._meanify_legacy(lu_min, lu_max, lv_min, lv_max)

    def _meanify_streaming(self):
        """Compute mean using streaming accumulators."""
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_average = self._grid_sum / self._grid_count
            mean_sq = self._grid_sum_sq / self._grid_count
            variance = mean_sq - (raw_average**2)
            raw_wrms = np.sqrt(np.maximum(variance, 0))

        # Match legacy output format
        # Transpose: legacy output is (n_v, n_u)
        self._average = raw_average.T
        self._wrms = raw_wrms.T

        # Centers
        u_centers = self._xedge[:-1] + self.dx / 2.0
        v_centers = self._yedge[:-1] + self.dy / 2.0

        # Meshgrid (legacy uses default 'xy' indexing -> returns (nv, nu))
        self._u0, self._v0 = np.meshgrid(u_centers, v_centers)

        # Sparse output
        count_T = self._grid_count.T
        valid_bins = (count_T > 0) & np.isfinite(self._average)

        self.coords0 = np.column_stack((self._u0[valid_bins], self._v0[valid_bins]))
        self.params0 = self._average[valid_bins]
        self.wrms0 = self._wrms[valid_bins]

    def _meanify_legacy(self, lu_min, lu_max, lv_min, lv_max):
        """Compute mean using legacy scipy-based approach."""
        params = np.concatenate(self.params)
        coords = np.concatenate(self.coords, axis=0)
        if self.stat_used == "weighted":
            params_err = np.concatenate(self.params_err)
            weights = 1.0 / params_err**2

        if lu_min is None:
            lu_min = np.min(coords[:, 0])
        if lu_max is None:
            lu_max = np.max(coords[:, 0])
        if lv_min is None:
            lv_min = np.min(coords[:, 1])
        if lv_max is None:
            lv_max = np.max(coords[:, 1])

        nbin_u = int((lu_max - lu_min) / self.bin_spacing)
        nbin_v = int((lv_max - lv_min) / self.bin_spacing)
        binning = [
            np.linspace(lu_min, lu_max, nbin_u),
            np.linspace(lv_min, lv_max, nbin_v),
        ]
        nbinning = (len(binning[0]) - 1) * (len(binning[1]) - 1)
        Filter = np.array([True] * nbinning)

        if self.stat_used == "weighted":
            sum_wpp, u0, v0, bin_target = binned_statistic_2d(
                coords[:, 0],
                coords[:, 1],
                weights * params * params,
                bins=binning,
                statistic="sum",
            )
            sum_wp, u0, v0, bin_target = binned_statistic_2d(
                coords[:, 0],
                coords[:, 1],
                weights * params,
                bins=binning,
                statistic="sum",
            )
            sum_w, u0, v0, bin_target = binned_statistic_2d(
                coords[:, 0], coords[:, 1], weights, bins=binning, statistic="sum"
            )
            average = sum_wp / sum_w
            wvar = (1.0 / sum_w) * (
                sum_wpp - 2.0 * average * sum_wp + average * average * sum_w
            )
            wrms = np.sqrt(wvar)
        else:
            if self.stat_used == "biweight":

                def stat(x):
                    return biweight(x, c=self.biweight_c)

            elif self.stat_used == "median_clipped":

                def stat(x):
                    return median_clipped(
                        x, nsigma=self.clip_nsigma, maxiters=self.clip_maxiters
                    )

            else:
                stat = self.stat_used
            average, xedge, yedge, bin_target = binned_statistic_2d(
                coords[:, 0],
                coords[:, 1],
                params,
                bins=binning,
                statistic=stat,
            )
            wrms = np.zeros_like(average)
        average = average.T
        wrms = wrms.T
        self._average = copy.deepcopy(average)
        self._wrms = wrms
        average = average.reshape(-1)
        wrms = wrms.reshape(-1)
        Filter &= np.isfinite(average).reshape(-1)
        Filter &= np.isfinite(wrms).reshape(-1)
        params0 = copy.deepcopy(average)
        u0 = copy.deepcopy(xedge)
        v0 = copy.deepcopy(yedge)
        wrms0 = wrms

        # get center of each bin
        u0 = u0[:-1] + (u0[1] - u0[0]) / 2.0
        v0 = v0[:-1] + (v0[1] - v0[0]) / 2.0
        u0, v0 = np.meshgrid(u0, v0)
        self._u0 = u0
        self._v0 = v0
        self._xedge = xedge
        self._yedge = yedge

        coords0 = np.array([u0.reshape(-1), v0.reshape(-1)]).T

        # remove any entries with nan (counts == 0 and non finite value in
        # the 2D statistic computation)
        self.coords0 = coords0[Filter]
        self.params0 = params0[Filter]
        self.wrms0 = wrms0[Filter]

    def save_results(self, name_output="mean_gp.fits"):
        """
        Write output mean function.

        :param name_output: Name of the output fits file. (default: 'mean_gp.fits')
        """
        dtypes = [
            ("COORDS0", self.coords0.dtype, self.coords0.shape),
            ("PARAMS0", self.params0.dtype, self.params0.shape),
            ("WRMS0", self.wrms0.dtype, self.wrms0.shape),
            ("_AVERAGE", self._average.dtype, self._average.shape),
            ("_WRMS", self._wrms.dtype, self._wrms.shape),
            ("_U0", self._u0.dtype, self._u0.shape),
            ("_V0", self._v0.dtype, self._v0.shape),
        ]
        data = np.empty(1, dtype=dtypes)

        data["COORDS0"] = self.coords0
        data["PARAMS0"] = self.params0
        data["WRMS0"] = self.wrms0
        data["_AVERAGE"] = self._average
        data["_WRMS"] = self._wrms
        data["_U0"] = self._u0
        data["_V0"] = self._v0

        with fitsio.FITS(name_output, "rw", clobber=True) as f:
            f.write_table(data, extname="average_solution")


# Backward compatibility alias
MeanifyStream = meanify
