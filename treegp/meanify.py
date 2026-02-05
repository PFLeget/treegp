"""
.. module:: meanify
"""

import numpy as np
from scipy.stats import binned_statistic_2d
import fitsio
import copy


class MeanifyStream(object):
    """
    High-performance, low-memory spatial averager. 
    
    This class uses an online/streaming algorithm. It pre-allocates the 
    spatial grid and accumulates statistics on the fly.
    Complexity: Time O(N), Memory O(1).
    """

    def __init__(self, bin_spacing=120.0, bounds=(0, 1, 0, 1)):
        """
        :param bin_spacing: Resolution of the grid.
        :param bounds: Tuple (u_min, u_max, v_min, v_max). 
                       REQUIRED for streaming mode to pre-allocate the grid.
        """
        if bounds is None:
            raise ValueError("Streaming Meanify requires 'bounds=(u_min, u_max, v_min, v_max)' at initialization.")
        
        self.bin_spacing = float(bin_spacing)
        self.lu_min, self.lu_max, self.lv_min, self.lv_max = bounds

        # 1. Define the grid dimensions immediately
        self.nbin_u = int((self.lu_max - self.lu_min) / self.bin_spacing)
        self.nbin_v = int((self.lv_max - self.lv_min) / self.bin_spacing)

        # 2. Pre-allocate accumulation maps (Y, X) convention for images usually, 
        # but we stick to (U, V) index order for consistency.
        shape = (self.nbin_u, self.nbin_v)
        
        # Accumulators
        self.grid_sum = np.zeros(shape, dtype=np.float64)    # Sum of values (Sigma x)
        self.grid_sum_sq = np.zeros(shape, dtype=np.float64) # Sum of squares (Sigma x^2)
        self.grid_count = np.zeros(shape, dtype=np.int64)    # Number of points (N)

        # Pre-compute grid centers for later export
        u_edges = np.linspace(self.lu_min, self.lu_max, self.nbin_u + 1)
        v_edges = np.linspace(self.lv_min, self.lv_max, self.nbin_v + 1)
        
        self.u_centers = u_edges[:-1] + (u_edges[1] - u_edges[0]) / 2.0
        self.v_centers = v_edges[:-1] + (v_edges[1] - v_edges[0]) / 2.0
        
        # We store these for consistency with original format
        self._xedge = u_edges
        self._yedge = v_edges

    def add_field(self, coord, param):
        """
        Add new data to the accumulator. 
        Does NOT store raw data in memory.
        
        :param coord: Array of coordinates (N, 2)
        :param param: Array of values (N,)
        """
        # 1. vectorized calculation of bin indices
        # We subtract min and divide by spacing to get integer index
        u_idx = np.floor((coord[:, 0] - self.lu_min) / self.bin_spacing).astype(int)
        v_idx = np.floor((coord[:, 1] - self.lv_min) / self.bin_spacing).astype(int)

        # 2. Filter points that are outside the pre-defined bounds
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

        # 3. Accumulate statistics using unbuffered addition
        # This handles cases where multiple points fall in the same bin within one batch
        np.add.at(self.grid_sum, (u_idx, v_idx), p)
        np.add.at(self.grid_sum_sq, (u_idx, v_idx), p**2)
        np.add.at(self.grid_count, (u_idx, v_idx), 1)

    def meanify(self):
        """
        Finalize the statistics. 
        Calculates Mean and RMS from the accumulated sums.
        """
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # 1. Calculate stats in (N_u, N_v) shape
            raw_average = self.grid_sum / self.grid_count
            
            mean_sq = self.grid_sum_sq / self.grid_count
            variance = mean_sq - (raw_average ** 2)
            variance = np.maximum(variance, 0)
            raw_wrms = np.sqrt(variance)

        # 2. MATCH LEGACY: Transpose to (N_v, N_u)
        # The legacy code does 'average = average.T', so we must too.
        self._average = raw_average.T
        self._wrms = raw_wrms.T
        
        # 3. MATCH LEGACY: Meshgrid
        # Legacy uses default np.meshgrid (Cartesian 'xy' indexing), which produces (N_v, N_u)
        # My previous code used 'ij' (Matrix) indexing. We switch to default to match legacy.
        u_grid, v_grid = np.meshgrid(self.u_centers, self.v_centers) # Default is indexing='xy'
        
        self._u0 = u_grid
        self._v0 = v_grid

        # 4. Filter for valid bins
        # Note: self.grid_count is still (N_u, N_v), so we transpose it to match _average
        count_T = self.grid_count.T
        valid_bins = (count_T > 0) & np.isfinite(self._average)

        # Flatten sparse representation
        self.coords0 = np.column_stack((
            u_grid[valid_bins], 
            v_grid[valid_bins]
        ))
        self.params0 = self._average[valid_bins]
        self.wrms0 = self._wrms[valid_bins]

    def save_results(self, name_output="mean_gp_stream.fits"):
        """
        Write output mean function to FITS.
        Matches the structure of the original class.
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
