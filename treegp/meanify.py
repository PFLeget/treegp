"""
.. module:: meanify
"""

from __future__ import print_function

import yaml
import os

def meanify(config):
    """Take Piff output(s), build an average of the FoV, and write output average.

    :param config:      The configuration file that defines how to build the model
    """
    from .star_temp import Star
    import glob
    import numpy as np
    from scipy.stats import binned_statistic_2d
    import fitsio

    for key in ['output', 'hyper']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)
    for key in ['file_name']:
        if key not in config['output']:
            raise ValueError("%s field is required in config dict output"%key)

    for key in ['file_name']:
        if key not in config['hyper']:
            raise ValueError("%s field is required in config dict hyper"%key)

    if 'dir' in config['output']:
        dir = config['output']['dir']
    else:
        dir = None

    if 'bin_spacing' in config['hyper']:
        bin_spacing = config['hyper']['bin_spacing'] #in arcsec
    else:
        bin_spacing = 120. #default bin_spacing: 120 arcsec

    if 'statistic' in config['hyper']:
        if config['hyper']['statistic'] not in ['mean', 'median']:
            raise ValueError("%s is not a suported statistic (only mean and median are currently suported)"
                             %config['hyper']['statistic'])
        else:
            stat_used = config['hyper']['statistic']
    else:
        stat_used = 'mean' #default statistics: arithmetic mean over each bin

    if 'params_fitted' in config['hyper']:
        if type(config['hyper']['params_fitted']) != list:
            raise TypeError('must give a list of index for params_fitted')
        else:
            params_fitted = config['hyper']['params_fitted']
    else:
        params_fitted = None

    if isinstance(config['output']['file_name'], list):
        psf_list = config['output']['file_name']
        if len(psf_list) == 0:
            raise ValueError("file_name may not be an empty list")
    elif isinstance(config['output']['file_name'], str):
        file_name = config['output']['file_name']
        if dir is not None:
            file_name = os.path.join(dir, file_name)
        psf_list = sorted(glob.glob(file_name))
        if len(psf_list) == 0:
            raise ValueError("No files found corresponding to "+config['file_name'])
    elif not isinstance(config['file_name'], dict):
        raise ValueError("file_name should be either a dict or a string")

    if psf_list is not None:
        npsfs = len(psf_list)
        config['output']['file_name'] = psf_list

    file_name_in = config['output']['file_name']

    file_name_out = config['hyper']['file_name']
    if 'dir' in config['hyper']:
        file_name_out = os.path.join(config['hyper']['dir'], file_name_out)

    coords = []
    params = []

    for fi, f in enumerate(file_name_in):
        fits = fitsio.FITS(f)
        coord, param = Star.read_coords_params(fits, 'psf_stars')
        fits.close()

        coords.append(coord)
        params.append(param)

    params = np.concatenate(params, axis=0)
    coords = np.concatenate(coords, axis=0)

    if params_fitted is None:
        params_fitted = range(len(params[0]))

    lu_min, lu_max = np.min(coords[:,0]), np.max(coords[:,0])
    lv_min, lv_max = np.min(coords[:,1]), np.max(coords[:,1])

    nbin_u = int((lu_max - lu_min) / bin_spacing)
    nbin_v = int((lv_max - lv_min) / bin_spacing)
    binning = [np.linspace(lu_min, lu_max, nbin_u), np.linspace(lv_min, lv_max, nbin_v)]
    nbinning = (len(binning[0]) - 1) * (len(binning[1]) - 1)
    params0 = np.zeros((nbinning, len(params[0])))
    Filter = np.array([True]*nbinning)

    for i in range(len(params[0])):
        if i in params_fitted:
            average, u0, v0, bin_target = binned_statistic_2d(coords[:,0], coords[:,1],
                                                              params[:,i], bins=binning,
                                                              statistic=stat_used)
            average = average.T
            average = average.reshape(-1)
            Filter &= np.isfinite(average).reshape(-1)
            params0[:,i] = average

    # get center of each bin 
    u0 = u0[:-1] + (u0[1] - u0[0])/2.
    v0 = v0[:-1] + (v0[1] - v0[0])/2.
    u0, v0 = np.meshgrid(u0, v0)

    coords0 = np.array([u0.reshape(-1), v0.reshape(-1)]).T

    # remove any entries with nan (counts == 0 and non finite value in
    # the 2D statistic computation) 
    coords0 = coords0[Filter]
    params0 = params0[Filter]

    dtypes = [('COORDS0', coords0.dtype, coords0.shape),
              ('PARAMS0', params0.dtype, params0.shape),
          ]
    data = np.empty(1, dtype=dtypes)

    data['COORDS0'] = coords0
    data['PARAMS0'] = params0

    with fitsio.FITS(file_name_out,'rw',clobber=True) as f:
        f.write_table(data, extname='average_solution')
