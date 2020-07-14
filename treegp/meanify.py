"""
.. module:: meanify
"""
import numpy as np
from scipy.stats import binned_statistic_2d
import fitsio
import os


class meanify(object):
    """Take data, build a spatial average, and write output average.

    :param bin_spacing: Bin_size, resolution on the mean function. (default=120.)
    :param statistics:  Statisitics used to compute the mean. (default=mean)
    """
    def __init__(self, bin_spacing=120., statistics='mean'):

        self.bin_spacing = bin_spacing

        if statistics not in ['mean', 'median']:
            raise ValueError("%s is not a suported statistic (only mean and median are currently suported)"
                             %(statistics))
        self.stat_used = statistics #default statistics: arithmetic mean over each bin

        self.coords = []
        self.params = []

    def add_field(self, coord, param):
        """
        Add new data to compute the mean function. 

        :param coord: Array of coordinate of the parameter.
        :param param: Array of parameter.
        """
        if np.shape(coord)[1] !=2:
            raise ValueError('meanify is supported only in 2d for the moment.')
        self.coords.append(coord)
        self.params.append(param)
    
    def meanify(self):
        """
        Compute the mean function.
        """
        params = np.concatenate(self.params)
        coords = np.concatenate(self.coords, axis=0)

        lu_min, lu_max = np.min(coords[:,0]), np.max(coords[:,0])
        lv_min, lv_max = np.min(coords[:,1]), np.max(coords[:,1])

        nbin_u = int((lu_max - lu_min) / self.bin_spacing)
        nbin_v = int((lv_max - lv_min) / self.bin_spacing)
        binning = [np.linspace(lu_min, lu_max, nbin_u), np.linspace(lv_min, lv_max, nbin_v)]
        nbinning = (len(binning[0]) - 1) * (len(binning[1]) - 1)
        Filter = np.array([True]*nbinning)

        average, u0, v0, bin_target = binned_statistic_2d(coords[:,0], coords[:,1],
                                                          params, bins=binning,
                                                          statistic=self.stat_used)
        average = average.T
        self._average = average
        self._u0 = u0
        self._v0 = v0
        average = average.reshape(-1)
        Filter &= np.isfinite(average).reshape(-1)
        params0 = average

        # get center of each bin 
        u0 = u0[:-1] + (u0[1] - u0[0])/2.
        v0 = v0[:-1] + (v0[1] - v0[0])/2.
        u0, v0 = np.meshgrid(u0, v0)

        coords0 = np.array([u0.reshape(-1), v0.reshape(-1)]).T

        # remove any entries with nan (counts == 0 and non finite value in
        # the 2D statistic computation) 
        self.coords0 = coords0[Filter]
        self.params0 = params0[Filter]

    def save_results(self, name_output='mean_gp.fits'):
        """
        Write output mean function.
        
        :param name_output: Name of the output fits file. (default: 'mean_gp.fits')
        """
        dtypes = [('COORDS0', self.coords0.dtype, self.coords0.shape),
                  ('PARAMS0', self.params0.dtype, self.params0.shape),
                  ('_AVERAGE', self._average.dtype, self._average.shape),
                  ('_U0', self.u0.dtype, self.u0.shape),
                  ('_V0', self.v0.dtype, self.v0.shape),
                  ]
        data = np.empty(1, dtype=dtypes)
        
        data['COORDS0'] = self.coords0
        data['PARAMS0'] = self.params0
        data['_AVERAGE'] = self._average
        data['_U0'] = self._u0
        data['_V0'] = self._v0

        with fitsio.FITS(name_output,'rw',clobber=True) as f:
            f.write_table(data, extname='average_solution')
