"""
.. module:: meanify
"""

import numpy as np
from scipy.stats import binned_statistic_2d
import fitsio
import copy


class MeanifyStream(object):
    def __init__(self, bin_spacing=120.0, bounds=None):
        """
        :param bin_spacing: Resolution of the grid.
        :param bounds: Tuple (u_min, u_max, v_min, v_max). 
        """
        if bounds is None:
            raise ValueError("Streaming Meanify requires bounds=(u_min, u_max, v_min, v_max)")
        
        self.bin_spacing = float(bin_spacing)
        self.lu_min, self.lu_max, self.lv_min, self.lv_max = bounds

        # --- REPLICATE LEGACY GRID LOGIC EXACTLY ---
        # Legacy: nbin_u = int((max - min) / spacing) is passed to linspace as num_points
        n_edges_u = int((self.lu_max - self.lu_min) / self.bin_spacing)
        n_edges_v = int((self.lv_max - self.lv_min) / self.bin_spacing)

        # Generate edges exactly like legacy
        self._xedge = np.linspace(self.lu_min, self.lu_max, n_edges_u)
        self._yedge = np.linspace(self.lv_min, self.lv_max, n_edges_v)

        # Actual number of bins is edges - 1
        self.nbin_u = len(self._xedge) - 1
        self.nbin_v = len(self._yedge) - 1

        # Calculate EFFECTIVE spacing for arithmetic binning
        # (This is crucial: linspace spacing != input bin_spacing due to integer rounding)
        self.dx = self._xedge[1] - self._xedge[0]
        self.dy = self._yedge[1] - self._yedge[0]

        # Accumulators (u, v) -> (x, y) order
        shape = (self.nbin_u, self.nbin_v)
        self.grid_sum = np.zeros(shape, dtype=np.float64)
        self.grid_sum_sq = np.zeros(shape, dtype=np.float64)
        self.grid_count = np.zeros(shape, dtype=np.int64)

    def add_field(self, coord, param):
        # 1. Arithmetic Binning using EFFECTIVE spacing
        # We use a small epsilon for stability, but rely primarily on clamping
        u_idx = ((coord[:, 0] - self.lu_min) / self.dx).astype(int)
        v_idx = ((coord[:, 1] - self.lv_min) / self.dy).astype(int)

        # 2. Handle Inclusive Right Edge (Match binned_statistic behavior)
        # If a point is exactly on the max edge, it belongs to the last bin, not the next one.
        on_u_edge = (coord[:, 0] == self.lu_max)
        on_v_edge = (coord[:, 1] == self.lv_max)
        u_idx[on_u_edge] = self.nbin_u - 1
        v_idx[on_v_edge] = self.nbin_v - 1

        # 3. Filter valid points
        valid_mask = (
            (u_idx >= 0) & (u_idx < self.nbin_u) & 
            (v_idx >= 0) & (v_idx < self.nbin_v) &
            np.isfinite(param)
        )

        if not np.any(valid_mask):
            return

        u_idx = u_idx[valid_mask]
        v_idx = v_idx[valid_mask]
        p = param[valid_mask]

        # 4. Accumulate
        np.add.at(self.grid_sum, (u_idx, v_idx), p)
        np.add.at(self.grid_sum_sq, (u_idx, v_idx), p**2)
        np.add.at(self.grid_count, (u_idx, v_idx), 1)

    def meanify(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_average = self.grid_sum / self.grid_count
            mean_sq = self.grid_sum_sq / self.grid_count
            variance = mean_sq - (raw_average ** 2)
            raw_wrms = np.sqrt(np.maximum(variance, 0))

        # --- MATCH LEGACY OUTPUT FORMAT ---
        
        # 1. Transpose: Legacy output is (n_v, n_u)
        self._average = raw_average.T
        self._wrms = raw_wrms.T
        
        # 2. Centers
        u_centers = self._xedge[:-1] + self.dx / 2.0
        v_centers = self._yedge[:-1] + self.dy / 2.0
        
        # 3. Meshgrid (Legacy uses default 'xy' indexing -> returns (nv, nu))
        self._u0, self._v0 = np.meshgrid(u_centers, v_centers)

        # 4. Sparse Output
        count_T = self.grid_count.T
        valid_bins = (count_T > 0) & np.isfinite(self._average)

        self.coords0 = np.column_stack((
            self._u0[valid_bins], 
            self._v0[valid_bins]
        ))
        self.params0 = self._average[valid_bins]
        self.wrms0 = self._wrms[valid_bins]

    def save_results(self, name_output="mean_gp.fits"):
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
            average, xedge, yedge, bin_target = binned_statistic_2d(
                coords[:, 0],
                coords[:, 1],
                params,
                bins=binning,
                statistic=self.stat_used,
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
