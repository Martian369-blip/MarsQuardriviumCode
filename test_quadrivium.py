import os
import numpy as np
from scipy.io.wavfile import read
from quadrivium_engine import (
    solve_kepler,
    compute_orbit_coordinates,
    generate_harmonic_audio,
    INNER_PLANETS,
)


def test_kepler_solver_circular_orbit():
    """For eccentricity e = 0, eccentric anomaly E must equal mean anomaly M."""
    M = np.linspace(0, 2 * np.pi, 50)
    E = solve_kepler(M, e=0.0)
    assert np.allclose(E, M, atol=1e-12)


def test_kepler_solver_elliptical_identity():
    """Verify M = E - e * sin(E) holds to machine precision for elliptical orbits."""
    M = np.array([0.0, 0.5, 1.5, np.pi, 5.0])
    e = 0.2056  # Mercury's eccentricity
    E = solve_kepler(M, e)
    reconstructed_M = E - e * np.sin(E)
    assert np.allclose(reconstructed_M, M, atol=1e-10)


def test_orbit_perihelion_and_aphelion():
    """Verify closest (perihelion) and farthest (aphelion) geometric distances."""
    a = 1.0
    e = 0.5
    # At M = 0 (perihelion), x should equal a * (1 - e) and y = 0
    x_peri, y_peri = compute_orbit_coordinates(a, e, 0.0)
    assert np.isclose(x_peri, a * (1 - e), atol=1e-10)
    assert np.isclose(y_peri, 0.0, atol=1e-10)

    # At M = pi (aphelion), x should equal -a * (1 + e) and y = 0
    x_aph, y_aph = compute_orbit_coordinates(a, e, np.pi)
    assert np.isclose(x_aph, -a * (1 + e), atol=1e-10)
    assert np.isclose(y_aph, 0.0, atol=1e-10)


def test_harmonic_audio_generation(tmp_path):
    """Ensure audio synthesis outputs a valid 16-bit PCM WAV file of expected length."""
    wav_file = tmp_path / "test_spheres.wav"
    duration = 1.0
    generate_harmonic_audio(INNER_PLANETS, base_freq=110.0, duration=duration, output_wav=str(wav_file))

    assert os.path.exists(wav_file)
    rate, data = read(str(wav_file))
    assert rate == 44100
    assert len(data) == 44100
    assert data.dtype == np.int16
