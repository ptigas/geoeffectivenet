import numpy as np
import pandas as pd
import torch
from scipy.special import sph_harm

r2d = 180 / np.pi
d2r = 1.0 / r2d


nm2ind = lambda n, m: n * n + n + m


def ind2nm(ind):
    n = int(np.floor(np.sqrt(ind)))
    m = ind - n * n - n
    return n, m


def basis_matrix(nmax, theta, phi):
    """Generate spherical harmonic basis on a grid.

    Args:
        nmax (int): Maximum number of modes (angular number l)
        theta (numpy array): 1D array of longitude
        phi (numpy array): 1D array of colatitude

    Returns:
        numpy array: 2D array of shape (l, (l+1)*2). For each l, we have 2*l+1 modes.
    """
    # theta is longitude, phi is colat
    assert len(theta) == len(phi)
    nbasis = (nmax + 1) * (nmax + 1) * 2  # 2 for real and imag components of Y_mn
    basis = np.zeros(shape=(len(theta), nbasis), dtype=np.float)
    for n in range(nmax + 1):
        for m in range(-n, n + 1):
            y_mn = sph_harm(m, n, theta, phi)
            basis[:, 2 * nm2ind(n, m)] = y_mn.real.ravel()
            basis[:, 2 * nm2ind(n, m) + 1] = y_mn.imag.ravel()
    return basis


""" function for computing subsolar point """


def subsol(datetimes):
    """
    calculate subsolar point at given datetime(s)

    Parameters
    ----------
    datetimes : datetime or list of datetimes
        datetime or list (or other iterable) of datetimes

    Returns
    -------
    subsol_lat : ndarray
        latitude(s) of the subsolar point
    subsol_lon : ndarray
        longiutde(s) of the subsolar point

    Note
    ----
    The code is vectorized, so it should be fast.

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac,
    results are good to at least 0.01 degree latitude and 0.025 degree
    longitude between years 1950 and 2050.  Accuracy for other years
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.
    """

    # use pandas DatetimeIndex for fast access to year, month day etc...
    if hasattr(datetimes, "__iter__"):
        datetimes = pd.DatetimeIndex(datetimes)
    else:
        datetimes = pd.DatetimeIndex([datetimes])

    year = np.float64(datetimes.year)
    # day of year:
    doy = np.float64(datetimes.dayofyear)
    # seconds since start of day:
    ut = np.float64(
        datetimes.hour * 60.0 ** 2 + datetimes.minute * 60.0 + datetimes.second
    )

    yr = year - 2000

    if year.max() >= 2100 or year.min() <= 1600:
        raise ValueError("subsol.py: subsol invalid after 2100 and before 1600")

    nleap = np.floor((year - 1601) / 4.0)
    nleap = np.array(nleap) - 99

    # exception for years <= 1900:
    ncent = np.floor((year - 1601) / 100.0)
    ncent = 3 - ncent
    nleap[year <= 1900] = nleap[year <= 1900] + ncent[year <= 1900]

    l0 = -79.549 + (-0.238699 * (yr - 4 * nleap) + 3.08514e-2 * nleap)

    g0 = -2.472 + (-0.2558905 * (yr - 4 * nleap) - 3.79617e-2 * nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut / 86400.0 - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = 0.9856474 * df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = 0.9856003 * df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g * np.pi / 180.0

    # Ecliptic longitude:
    lmbda = l + 1.915 * np.sin(grad) + 0.020 * np.sin(2.0 * grad)
    lmrad = lmbda * np.pi / 180.0
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.0 * yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.0e-7 * n
    epsrad = epsilon * np.pi / 180.0

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad) * sinlm, np.cos(lmrad)) * 180.0 / np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad) * sinlm) * 180.0 / np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg / 360.0)
    etdeg = etdeg - 360.0 * nrot

    # Apparent time (degrees):
    aptime = ut / 240.0 + etdeg  # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180.0 - aptime
    nrot = np.round(sbsllon / 360.0)
    sbsllon = sbsllon - 360.0 * nrot

    return sbsllat, sbsllon


def is_leapyear(year):
    """Return True if leapyear else False

    Handles arrays and preserves shape

    Parameters
    ----------
    year : array_like
        array of years

    Returns
    -------
    is_leapyear : ndarray of bools
        True where input is leapyear, False elsewhere
    """

    # if array:
    if type(year) is np.ndarray:
        out = np.full_like(year, False, dtype=bool)

        out[year % 4 == 0] = True
        out[year % 100 == 0] = False
        out[year % 400 == 0] = True

        return out

    # if scalar:
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def sza(glat, glon, time):
    """return solar zenith angle - the angle of a vertical axis with the Sun Earth line -
    at given geographic latitudes, longitudes, and times

    input and output in degrees
    """

    # calculate subsolar points
    sslat, sslon = subsol(time)

    ssr = sph_to_car(np.vstack((np.ones_like(sslat), 90 - sslat, 90 - sslon)), deg=True)
    gcr = sph_to_car(np.array([[1.0], [90 - glat], [glon]]), deg=True)

    # the angle is arccos of the dot product of these two vectors
    return np.arccos(np.sum(ssr * gcr, axis=0)) * 180 / np.pi


# first make arrays of IGRF dipole coefficients. This is used to make rotation matrix from geo to cd coords
# these values are from https://www.ngdc.noaa.gov/IAGA/vmod/igrf12coeffs.txt
time = [
    1900.0,
    1905.0,
    1910.0,
    1915.0,
    1920.0,
    1925.0,
    1930.0,
    1935.0,
    1940.0,
    1945.0,
    1950.0,
    1955.0,
    1960.0,
    1965.0,
    1970.0,
    1975.0,
    1980.0,
    1985.0,
    1990.0,
    1995.0,
    2000.0,
    2005.0,
    2010.0,
    2015.0,
    2020.0,
    2025.0,
]
g10 = [
    -31543,
    -31464,
    -31354,
    -31212,
    -31060,
    -30926,
    -30805,
    -30715,
    -30654,
    -30594,
    -30554,
    -30500,
    -30421,
    -30334,
    -30220,
    -30100,
    -29992,
    -29873,
    -29775,
    -29692,
    -29619.4,
    -29554.63,
    -29496.57,
    -29441.46,
    -29404.8,
]
g11 = [
    -2298,
    -2298,
    -2297,
    -2306,
    -2317,
    -2318,
    -2316,
    -2306,
    -2292,
    -2285,
    -2250,
    -2215,
    -2169,
    -2119,
    -2068,
    -2013,
    -1956,
    -1905,
    -1848,
    -1784,
    -1728.2,
    -1669.05,
    -1586.42,
    -1501.77,
    -1450.9,
]
h11 = [
    5922,
    5909,
    5898,
    5875,
    5845,
    5817,
    5808,
    5812,
    5821,
    5810,
    5815,
    5820,
    5791,
    5776,
    5737,
    5675,
    5604,
    5500,
    5406,
    5306,
    5186.1,
    5077.99,
    4944.26,
    4795.99,
    4652.5,
]
g10sv = 5.7  # secular variations
g11sv = 7.4
h11sv = -25.9
g10.append(
    g10[-1] + g10sv * 5
)  # append 2020 values using secular variation coefficients
g11.append(g11[-1] + g11sv * 5)
h11.append(h11[-1] + h11sv * 5)
igrf_dipole = pd.DataFrame(
    {"g10": np.array(g10), "g11": np.array(g11), "h11": np.array(h11)}, index=time
)
igrf_dipole["B0"] = np.sqrt(
    igrf_dipole["g10"] ** 2 + igrf_dipole["g11"] ** 2 + igrf_dipole["h11"] ** 2
)


def dipole_axis(epoch):
    """calculate dipole axis in geocentric ECEF coordinates for given epoch(s)

    Calculations are based on IGRF coefficients, and linear interpolation is used
    in between IGRF models (defined every 5 years). Secular variation coefficients
    are used for the five years after the latest model.

    Parameters
    ----------
    epoch : float or array of floats
        year (with fraction) for which the dipole axis will be calculated. Multiple
        epochs can be given, as an array of N floats, resulting in a N x 3-dimensional
        return value

    Returns
    -------
    axes : array
        N x 3-dimensional array, where N is the number of inputs (epochs), and the
        columns contain the x, y, and z components of the corresponding dipole axes

    """

    epoch = np.asarray(
        epoch
    ).flatten()  # turn input into array in case it isn't already

    # interpolate Gauss coefficients to the input times:
    dipole = (
        igrf_dipole.reindex(list(igrf_dipole.index) + list(epoch))
        .sort_index()
        .interpolate()
        .drop_duplicates()
    )

    params = {key: dipole.loc[epoch, key].values for key in ["g10", "g11", "h11", "B0"]}

    Z_cd = -np.vstack((params["g11"], params["h11"], params["g10"])) / params["B0"]

    return Z_cd.T


def dipole_poles(epoch):
    """calculate dipole pole positions at given epoch(s)

    Parameters
    ----------
    epoch : float or array of floats
        year (with fraction) for which the dipole axis will be calculated. Multiple
        epochs can be given, as an array of N floats, resulting in a N x 3-dimensional
        return value

    Returns
    -------
    north_colat : array
        colatitude of the dipole pole in the northern hemisphere, same number of
        values as input
    north_longitude: array
        longitude of the dipole pole in the northern hemisphere, same number of
        values as input
    south_colat : array
        colatitude of the dipole pole in the southern hemisphere, same number of
        values as input
    south_longitude: array
        longitude of the dipole pole in the southern hemisphere, same number of
        values as input




    """
    print(dipole_axis(epoch))
    north_colat, north_longitude = car_to_sph(dipole_axis(epoch).T, deg=True)[1:]
    south_colat, south_longitude = car_to_sph(-dipole_axis(epoch).T, deg=True)[1:]

    return north_colat, north_longitude, south_colat, south_longitude


def geo2mag(glat, glon, Ae=None, An=None, epoch=2020, deg=True, inverse=False):
    """Convert geographic (geocentric) to centered dipole coordinates

    The conversion uses IGRF coefficients directly, interpolated
    to the provided epoch. The construction of the rotation matrix
    follows Laundal & Richmond (2017) [4]_ .

    Preserves shape. glat, glon, Ae, and An should have matching shapes

    Parameters
    ----------
    glat : array_like
        array of geographic latitudes
    glon : array_like
        array of geographic longitudes
    Ae   : array-like, optional
        array of eastward vector components to be converted. Default
        is 'none', and no converted vector components will be returned
    An   : array-like, optional
        array of northtward vector components to be converted. Default
        is 'none', and no converted vector components will be returned
    epoch : float, optional
        epoch (year) for the dipole used in the conversion, default 2020
    deg : bool, optional
        True if input is in degrees, False otherwise
    inverse: bool, optional
        set to True to convert from magnetic to geographic.
        Default is False

    Returns
    -------
    cdlat : ndarray
        array of centered dipole latitudes [degrees]
    cdlon : ndarray
        array of centered dipole longitudes [degrees]
    Ae_cd : ndarray
        array of eastward vector components in dipole coords
        (if Ae != None and An != None)
    An_cd : ndarray
        array of northward vector components in dipole coords
        (if Ae != None and An != None)

    """

    shape = np.asarray(glat).shape
    glat, glon = np.asarray(glat).flatten(), np.asarray(glon).flatten()

    # Find IGRF parameters for given epoch:
    dipole = (
        igrf_dipole.reindex(list(igrf_dipole.index) + [epoch])
        .sort_index()
        .interpolate()
        .drop_duplicates()
    )
    dipole = dipole.loc[epoch, :]

    # make rotation matrix from geo to cd
    Zcd = -np.array([dipole.g11, dipole.h11, dipole.g10]) / dipole.B0
    Zgeo_x_Zcd = np.cross(np.array([0, 0, 1]), Zcd)
    Ycd = Zgeo_x_Zcd / np.linalg.norm(Zgeo_x_Zcd)
    Xcd = np.cross(Ycd, Zcd)

    Rgeo_to_cd = np.vstack((Xcd, Ycd, Zcd))

    if inverse:  # transpose rotation matrix to get inverse operation
        Rgeo_to_cd = Rgeo_to_cd.T

    # convert input to ECEF:
    colat = 90 - glat.flatten() if deg else np.pi / 2 - glat.flatten()
    glon = glon.flatten()
    r_geo = sph_to_car(np.vstack((np.ones_like(colat), colat, glon)), deg=deg)

    # rotate:
    r_cd = Rgeo_to_cd.dot(r_geo)

    # convert result back to spherical:
    _, colat_cd, lon_cd = car_to_sph(r_cd, deg=True)

    # return coords if vector components are not to be calculated
    if any([Ae is None, An is None]):
        return 90 - colat_cd.reshape(shape), lon_cd.reshape(shape)

    Ae, An = np.asarray(Ae).flatten(), np.asarray(An).flatten()
    A_geo_enu = np.vstack((Ae, An, np.zeros(Ae.size)))
    A = np.sqrt(Ae ** 2 + An ** 2)
    A_geo_ecef = enu_to_ecef(
        (A_geo_enu / A).T, glon, glat
    )  # rotate normalized vectors to ecef
    A_cd_ecef = Rgeo_to_cd.dot(A_geo_ecef.T)
    A_cd_enu = ecef_to_enu(A_cd_ecef.T, lon_cd, 90 - colat_cd).T * A

    # return coords and vector components:
    return (
        90 - colat_cd.reshape(shape),
        lon_cd.reshape(shape),
        A_cd_enu[0].reshape(shape),
        A_cd_enu[1].reshape(shape),
    )


def dipole_tilt(times, epoch=2015.0):
    """Calculate dipole tilt angle for given set of times, at given epoch(s)

    Parameters
    ----------
    times : datetime or array/list of datetimes
        Times for which the dipole tilt angle should be calculated
    epoch : float or array/list of floats, optional
        Year (with fraction) for calculation of dipole axis. This should either be
        a scalar, or contain as man elements as times. Default epoch is 2015.0

    Return
    ------
    tilt_angle : array
        Array of dipole tilt angles in degrees

    Example
    -------
    >>> from datetime import datetime

    >>> print dipole_tilt(datetime(1927, 6, 10, 12, 00), epoch = 1927)
    [ 26.79107107]

    >>> # several times can be given. If they are close in time, one epoch should be fine
    >>> print dipole_tilt([datetime(1927, 6, 10, 12, 00), datetime(1927, 6, 10, 10, 00)], epoch = 1927)
    [ 26.79107107  20.89550663]

    >>> # if the times are far apart, the epoch for each can be specified to take field changes into account
    >>> # this will be a bit slower if many times are given
    >>> print dipole_tilt([datetime(1927, 6, 10, 12, 00), datetime(2015, 6, 10, 10, 00)], epoch = (1927, 2015))
    [ 26.79107107  20.59137527]

    """

    epoch = np.squeeze(np.array(epoch)).flatten()
    times = np.squeeze(np.array(times)).flatten()

    if not (epoch.shape == times.shape or epoch.shape == (1,)):
        raise ValueError(
            "epoch should either be scalar or have as many elements as times"
        )

    # get subsolar point coordinates
    sslat, sslon = subsol(times)

    s = np.vstack(
        (
            np.cos(sslat * d2r) * np.cos(sslon * d2r),
            np.cos(sslat * d2r) * np.sin(sslon * d2r),
            np.sin(sslat * d2r),
        )
    ).T
    m = dipole_axis(epoch)

    # calculate tilt angle:
    return np.arcsin(np.sum(s * m, axis=1)) * r2d


def R2(true, pred):
    mse = ((true-pred)**2).mean()
    var = (true-pred).var()
    return 1 - mse/var