import numpy as np

TUNING_SYSTEMS = {
    "Pythagorean (3-Limit Ratios)": {
        "Unison (1:1)": 1.0,
        "Minor Third (32:27)": 32.0 / 27.0,
        "Major Third (81:64)": 81.0 / 64.0,
        "Perfect Fourth (4:3)": 4.0 / 3.0,
        "Perfect Fifth (3:2)": 3.0 / 2.0,
        "Major Sixth (27:16)": 27.0 / 16.0,
        "Octave (2:1)": 2.0,
    },
    "Just Intonation (5-Limit Pure)": {
        "Unison (1:1)": 1.0,
        "Minor Third (6:5)": 6.0 / 5.0,
        "Major Third (5:4)": 5.0 / 4.0,
        "Perfect Fourth (4:3)": 4.0 / 3.0,
        "Perfect Fifth (3:2)": 3.0 / 2.0,
        "Major Sixth (5:3)": 5.0 / 3.0,
        "Octave (2:1)": 2.0,
    },
    "Equal Temperament (12-TET)": {
        "Unison": 2.0 ** (0 / 12.0),
        "Minor Third": 2.0 ** (3 / 12.0),
        "Major Third": 2.0 ** (4 / 12.0),
        "Perfect Fourth": 2.0 ** (5 / 12.0),
        "Perfect Fifth": 2.0 ** (7 / 12.0),
        "Major Sixth": 2.0 ** (9 / 12.0),
        "Octave": 2.0 ** (12 / 12.0),
    },
}


def compute_fft_spectrum(signal, sample_rate):
    """Computes Fast Fourier Transform (FFT) frequency magnitudes."""
    n = len(signal)
    yf = np.fft.rfft(signal)
    xf = np.fft.rfftfreq(n, 1.0 / sample_rate)
    magnitude = np.abs(yf) / n
    return xf, magnitude


def generate_chladni_plate(n, m, resolution=250):
    """Computes 2D Chladni acoustic nodal standing wave patterns on a square plate."""
    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(n * np.pi * X) * np.cos(m * np.pi * Y) - np.cos(m * np.pi * X) * np.cos(n * np.pi * Y)
    return X, Y, Z
