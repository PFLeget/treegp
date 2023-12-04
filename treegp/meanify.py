"""
.. module:: meanify
"""
import numpy as np
from scipy.stats import binned_statistic_2d
import fitsio


class meanify(object):
    """Take data, build a spatial average, and write output average.

    :param bin_spacing: Bin_size, resolution on the mean function. (default=120.)
    :param statistics:  Statisitics used to compute the mean. (default=mean)
    """

    def __init__(self, bin_spacing=120.0, statistics="mean"):
        self.bin_spacing = bin_spacing

        if statistics not in ["mean", "median", "weighted"]:
            raise ValueError(
                "%s is not a suported statistic (only mean, weighted, and median are currently suported)"
                % (statistics)
            )
        self.stat_used = statistics  # default statistics: arithmetic mean over each bin

        self.coords = []
        self.params = []
        self.params_err = []

    def add_field(self, coord, param, params_err=None):
        """
        Add new data to compute the mean function.

        :param coord: Array of coordinate of the parameter.
        :param param: Array of parameter.
        """
        if np.shape(coord)[1] != 2:
            raise ValueError("meanify is supported only in 2d for the moment.")
        self.coords.append(coord)
        self.params.append(param)
        if self.stat_used == "weighted":
            if params_err is None:
                raise ValueError("Need an associated error to params")
            else:
                self.params_err.append(params_err)

    def meanify(self, lu_min=None, lu_max=None, lv_min=None, lv_max=None):
        """
        Compute the mean function.
        """
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
            average, u0, v0, bin_target = binned_statistic_2d(
                coords[:, 0],
                coords[:, 1],
                params,
                bins=binning,
                statistic=self.stat_used,
            )
            wrms = np.zeros_like(average)
        average = average.T
        wrms = wrms.T
        self._average = average
        self._wrms = wrms
        average = average.reshape(-1)
        wrms = wrms.reshape(-1)
        Filter &= np.isfinite(average).reshape(-1)
        Filter &= np.isfinite(wrms).reshape(-1)
        params0 = average
        wrms0 = wrms

        # get center of each bin
        u0 = u0[:-1] + (u0[1] - u0[0]) / 2.0
        v0 = v0[:-1] + (v0[1] - v0[0]) / 2.0
        u0, v0 = np.meshgrid(u0, v0)
        self._u0 = u0
        self._v0 = v0

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
