"""
.. module:: meanify_healpix
"""

import numpy as np
import fitsio


class meanify_healpix:
    """Compute spatial averages on the sphere using HEALPix binning.

    This class mirrors the meanify API but uses healsparse for spherical
    binning, properly handling spherical geometry for sky coordinates (RA/Dec).

    :param bin_spacing: Approximate bin size in arcsec (converted to nside).
                        Default is 120 arcsec.
    :param nside: HEALPix nside parameter. If provided, overrides bin_spacing.
                  Must be a power of 2.
    """

    def __init__(self, bin_spacing=120.0, nside=None):
        import healsparse as hsp

        if nside is not None:
            self.nside = int(nside)
            # Verify nside is a power of 2
            if self.nside & (self.nside - 1) != 0 or self.nside < 1:
                raise ValueError("nside must be a positive power of 2")
        else:
            # Convert bin_spacing (arcsec) to nside
            # HEALPix pixel angular size: theta = sqrt(4*pi / (12*nside^2)) radians
            # Solving for nside: nside = sqrt(4*pi/12) * (206265 / bin_spacing)
            nside_approx = np.sqrt(4 * np.pi / 12) * 206265.0 / bin_spacing
            # Round to nearest power of 2
            self.nside = 2 ** int(np.round(np.log2(nside_approx)))
            # Ensure minimum nside of 1
            self.nside = max(1, self.nside)

        self.bin_spacing = bin_spacing

        # Compute actual pixel size for reference
        self.pixel_area_sr = 4 * np.pi / (12 * self.nside**2)
        self.pixel_size_arcsec = np.sqrt(self.pixel_area_sr) * 206265.0

        # Initialize healsparse accumulators using WIDE_MASK dtype
        # We need 3 fields: sum, sum_sq, count
        self._sum_map = hsp.HealSparseMap.make_empty(
            nside_coverage=32,
            nside_sparse=self.nside,
            dtype=np.float64,
            sentinel=0.0,
        )
        self._sum_sq_map = hsp.HealSparseMap.make_empty(
            nside_coverage=32,
            nside_sparse=self.nside,
            dtype=np.float64,
            sentinel=0.0,
        )
        self._count_map = hsp.HealSparseMap.make_empty(
            nside_coverage=32,
            nside_sparse=self.nside,
            dtype=np.int64,
            sentinel=0,
        )

    def add_field(self, coord, param):
        """Add new data to compute the mean function.

        :param coord: Array of shape (N, 2) with [RA, Dec] in degrees.
        :param param: Array of parameter values of length N.
        """
        import hpgeom as hpg

        coord = np.asarray(coord)
        param = np.asarray(param)

        if coord.ndim != 2 or coord.shape[1] != 2:
            raise ValueError("coord must have shape (N, 2) with [RA, Dec] columns")

        if len(param) != len(coord):
            raise ValueError("param must have same length as coord")

        # Filter out non-finite values
        valid_mask = np.isfinite(param)
        if not np.any(valid_mask):
            return

        ra = coord[valid_mask, 0]
        dec = coord[valid_mask, 1]
        p = param[valid_mask]

        # Convert RA/Dec to HEALPix pixel indices using hpgeom
        pixels = hpg.angle_to_pixel(self.nside, ra, dec, nest=True, degrees=True)

        # Accumulate values per pixel
        unique_pixels, inverse = np.unique(pixels, return_inverse=True)

        # Sum values per pixel
        pixel_sums = np.bincount(inverse, weights=p, minlength=len(unique_pixels))
        pixel_sum_sqs = np.bincount(
            inverse, weights=p**2, minlength=len(unique_pixels)
        )
        pixel_counts = np.bincount(inverse, minlength=len(unique_pixels))

        # Update healsparse maps
        for i, pix in enumerate(unique_pixels):
            self._sum_map[pix] = self._sum_map[pix] + pixel_sums[i]
            self._sum_sq_map[pix] = self._sum_sq_map[pix] + pixel_sum_sqs[i]
            self._count_map[pix] = self._count_map[pix] + pixel_counts[i]

    def meanify(self):
        """Compute the mean function from accumulated data.

        Populates coords0, params0, and wrms0 attributes.
        """
        import hpgeom as hpg

        # Get valid pixels (those with data)
        valid_pixels = self._count_map.valid_pixels

        if len(valid_pixels) == 0:
            self.coords0 = np.empty((0, 2))
            self.params0 = np.empty(0)
            self.wrms0 = np.empty(0)
            return

        # Extract accumulated values
        sums = np.array([self._sum_map[pix] for pix in valid_pixels])
        sum_sqs = np.array([self._sum_sq_map[pix] for pix in valid_pixels])
        counts = np.array([self._count_map[pix] for pix in valid_pixels])

        # Compute mean and RMS
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
            mean_sqs = sum_sqs / counts
            variance = mean_sqs - means**2
            wrms = np.sqrt(np.maximum(variance, 0))

        # Get pixel centers in RA/Dec using hpgeom
        ra, dec = hpg.pixel_to_angle(self.nside, valid_pixels, nest=True, degrees=True)

        # Filter out any remaining non-finite values
        valid_mask = np.isfinite(means) & np.isfinite(wrms)

        self.coords0 = np.column_stack((ra[valid_mask], dec[valid_mask]))
        self.params0 = means[valid_mask]
        self.wrms0 = wrms[valid_mask]

        # Store full arrays for potential grid access
        self._valid_pixels = valid_pixels[valid_mask]
        self._means = means[valid_mask]
        self._wrms = wrms[valid_mask]

    def save_results(self, name_output="mean_gp_healpix.fits"):
        """Write output mean function to FITS file.

        :param name_output: Name of the output FITS file.
                            Default is 'mean_gp_healpix.fits'.
        """
        dtypes = [
            ("COORDS0", self.coords0.dtype, self.coords0.shape),
            ("PARAMS0", self.params0.dtype, self.params0.shape),
            ("WRMS0", self.wrms0.dtype, self.wrms0.shape),
            ("NSIDE", np.int32, ()),
            ("PIXEL_SIZE_ARCSEC", np.float64, ()),
        ]
        data = np.empty(1, dtype=dtypes)

        data["COORDS0"] = self.coords0
        data["PARAMS0"] = self.params0
        data["WRMS0"] = self.wrms0
        data["NSIDE"] = self.nside
        data["PIXEL_SIZE_ARCSEC"] = self.pixel_size_arcsec

        with fitsio.FITS(name_output, "rw", clobber=True) as f:
            f.write_table(data, extname="average_solution")
