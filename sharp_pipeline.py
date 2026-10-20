# Calculation of magnetic features from SHARP vector magnetograms, following Bobra & Couvidat (2015).
# Created by Suvadip Sinha, Modified by Nabdeep Patel
# Updated on : 2026-10-20

import numpy as np
from astropy.io import fits

# defining the constants
MU0 = 4 * np.pi * 1e-7 # in tesla meter per ampere
GAUSS_TO_TESLA = 1e-4
RAD = np.pi / 180.

SHARP_KEYS = [
    'T_REC', 'HARPNUM', 'NOAA_AR', 'USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ',
    'MEANGBH', 'MEANJZD', 'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH',
    'SAVNCPP', 'MEANPOT', 'TOTPOT', 'MEANSHR', 'SHRGT45', 'R_VALUE', 'AREA_ACR',
    'MSKEW', 'MKURT', 'CODEVER7', 'QUALITY', 'CDELT1', 'RSUN_REF', 'RSUN_OBS', 'DSUN_OBS',
]


class SharpHeader:
    def __init__(self, fits_file):
        with fits.open(fits_file) as hdul:
            header = hdul[1].header
            self.data = {key: header[key] for key in SHARP_KEYS}

        d = self.data
        self.cdelt_arcsec = np.arctan(d['CDELT1'] * RAD * d['RSUN_REF'] / d['DSUN_OBS']) / RAD * 3600.


class MagneticField:
    # Components (HMI -> local Cartesian):  Bx = Bp,  By = -Bt,  Bz = Br   [Gauss]
    def __init__(self, br, bt, bp, conf_disambig, bitmap, header):
        self.bx, self.by, self.bz = bp, -bt, br
        self.rsun_ref = header['RSUN_REF']
        self.rsun_obs = header['RSUN_OBS']
        self.dsun_obs = header['DSUN_OBS']
        self.cdelt_arcsec = np.arctan(header['CDELT1'] * RAD * self.rsun_ref / self.dsun_obs) / RAD * 3600.

        # Valid pixels: well disambiguated, inside the active-region patch, no NaNs
        self.valid = ((conf_disambig >= 70) & (bitmap >= 30)
                      & ~np.isnan(self.bx) & ~np.isnan(self.by) & ~np.isnan(self.bz))

        # Physical size of one pixel
        self.pix_m = self.cdelt_arcsec * self.rsun_ref / self.rsun_obs      # metres
        self.pix_cm = self.cdelt_arcsec / 3600. * RAD * self.dsun_obs * 100.  # centimetres

    # ---------- vertical current ----------
    def vertical_current(self):
        """Jz = dBy/dx - dBx/dy (in Gauss/pixel), plus summary currents.

        Returns: jz_gpix, net_current [mA/m^2 sum], total_unsigned_current [A], mean_jz [mA/m^2]
        """
        dby_dx = np.gradient(self.by, axis=1, edge_order=2)
        dbx_dy = np.gradient(self.bx, axis=0, edge_order=2)
        jz = dby_dx - dbx_dy
        jz[[0, -1], :] = 0      # edge rows/columns are not used 
        jz[:, [0, -1]] = 0

        jz_valid = jz[self.valid]
        # Gauss/pixel -> A/m^2:  (1e-4 T/G) / (MU0) / (pixel size in m)
        to_amp_per_m2 = GAUSS_TO_TESLA / MU0 / self.pix_m

        net = jz_valid.sum() * to_amp_per_m2 * 1e3                          # mA/m^2
        total_unsigned = np.abs(jz_valid).sum() * to_amp_per_m2 * self.pix_m ** 2  # A
        mean_jz = net / jz_valid.size
        return jz, net, total_unsigned, mean_jz

    # ---------- twist ----------
    def alpha(self, jz, min_bz=0.):
        """Force-free parameter alpha = Jz / Bz [1/Mm], for valid pixels with |Bz| >= min_bz."""
        keep = self.valid & (jz != 0) & (np.abs(self.bz) >= min_bz) & (self.bz != 0)
        return jz[keep] / self.bz[keep] / self.pix_m * 1e6

    def twist_features(self, jz):
        """Returns: (std_twist, total_abs_twist, top10_mean_twist)."""
        # Spread of alpha over all valid pixels (scaled by MU0, as in the original code)
        std_twist = MU0 * self.alpha(jz).std()

        # Hahn twist: only strong-field pixels (|Bz| >= 300 G)
        abs_twist = np.abs(self.alpha(jz, min_bz=300.))
        n_top = max(1, round(0.1 * abs_twist.size))
        top10_mean = np.sort(abs_twist)[-n_top:].mean()
        return std_twist, abs_twist.sum(), top10_mean

    # ---------- Lorentz force ----------
    def lorentz_force(self):
        """Returns: total_fz [dyne], total_B^2 [G^2], epsilon_z, total_lorentz [dyne]."""
        bx, by, bz = self.bx[self.valid], self.by[self.valid], self.bz[self.valid]
        area = self.pix_cm ** 2

        fr = np.sum(bx**2 + by**2 - bz**2)
        bsq = np.sum(bx**2 + by**2 + bz**2)
        fx = area * np.sum(bz * bx) / (4 * np.pi)
        fy = area * np.sum(bz * by) / (4 * np.pi)
        fz = area * fr / (8 * np.pi)

        total_force = np.sqrt(fx**2 + fy**2 + fz**2)
        eps_z = fr / bsq if bsq > 0 else np.nan
        return fz, bsq, eps_z, total_force


def main(br, bt, bp, conf_disambig, bitmap, fits_file):
    header = SharpHeader(fits_file).data
    field = MagneticField(br, bt, bp, conf_disambig, bitmap, header)

    jz, net_current, total_unsigned_current, mean_jz = field.vertical_current()
    std_twist, total_abs_twist, top10_twist = field.twist_features(jz)
    total_fz, total_bsq, eps_z, _ = field.lorentz_force()

    computed = {
        'CDELT_ARC': header.cdelt_arcsec,
        'NET_CURRENT': net_current,
        'TOT_UNSIGNED_CURRENT': total_unsigned_current,
        'MEAN_JZ': mean_jz,
        'STD_TWIST': std_twist,
        'TOT_ABS_TWIST': total_abs_twist,
        'TOP10_TWIST': top10_twist,
        'EPSZ': eps_z,
        'TOTBSQ': total_bsq,
        'TOTFZ': total_fz,
    }

    return {**header, **computed}
