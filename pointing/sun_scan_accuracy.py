"""
This script computes the maximum angular error introduced when performing
a linear scan in azimuth and elevation (Az/El) between two endpoints
(A and B), compared to the true trajectory of the Sun in the sky, based
on accurate RA/Dec-to-AltAz conversions over time.

It simulates five 15-second scans of the Sun observed from Concordia
Station (Antarctica) on December 21, 2025, between 00:00 and 12:00 UTC
(local time is UTC+8). It samples the Sun's true position every 0.5 seconds,
converts each to AltAz, and compares it to a linear interpolation between
the starting and ending Az/El points.

It reports the maximum angular difference in arcseconds, based
 on the five scans"""

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time

# Concordia Station
location = EarthLocation(lat=-75.1 * u.deg, lon=123.35 * u.deg, height=3233 * u.m)

# Times: 21 Dec 2025, from 00:00 to 12:00 UTC
# Concordia local time is UTC+8
start_times = (
    Time("2025-12-21 00:00:00", scale="utc", location=location),
    Time("2025-12-21 03:00:00", scale="utc", location=location),
    Time("2025-12-21 06:00:00", scale="utc", location=location),
    Time("2025-12-21 09:00:00", scale="utc", location=location),
    Time("2025-12-21 12:00:00", scale="utc", location=location),
)
error = 0 * u.arcsec

for start_time in start_times:
    duration = 15 * u.s  # scan duration (seconds)
    num_points = 31  # sample every 0.5s
    times = start_time + np.linspace(0, duration.value, num_points) * u.s

    # Get true solar positions in Az/El, accounting for atmospheric
    # refraction (based on pressure, temperature, humidity, wavelength).
    altaz_frame = AltAz(
        obstime=times,
        location=location,
        pressure=650 * u.hPa,
        temperature=-30 * u.deg_C,
        relative_humidity=0.2,
        obswl=3 * u.mm  # wavelength for 100 GHz
    )
    sun_altaz = get_sun(times).transform_to(altaz_frame)

    # Initial and final points
    az_A, el_A = sun_altaz.az[0], sun_altaz.alt[0]
    az_B, el_B = sun_altaz.az[-1], sun_altaz.alt[-1]

    # Linear interpolation between A and B
    linear_az = az_A + (az_B - az_A) * np.linspace(0, 1, num_points)
    linear_el = el_A + (el_B - el_A) * np.linspace(0, 1, num_points)

    # Convert all to radians explicitly (scalar quantities)
    az_true = sun_altaz.az.to(u.rad).value
    el_true = sun_altaz.alt.to(u.rad).value
    az_lin = linear_az.to(u.rad).value
    el_lin = linear_el.to(u.rad).value

    # Compute angular separation point by point
    def angular_distance(az1, el1, az2, el2):
        # spherical angular distance formula (haversine-like for sphere)
        return np.arccos(
            np.sin(el1) * np.sin(el2) + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
        )

    errors_rad = angular_distance(az_true, el_true, az_lin, el_lin)
    errors_arcsec = (errors_rad * u.rad).to(u.arcsec)
    if np.max(errors_arcsec) > error:
        error = np.max(errors_arcsec)
        t = start_time

# Show maximum error
print(f"Maximum angular error: {error:.3f} - {t}")
