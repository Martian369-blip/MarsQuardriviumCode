import io
import json
import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from scipy.linalg import eigh_tridiagonal
import streamlit as st
import sympy as sp

# ==============================================================================
# 1. MATHEMATICS MODULE
# ==============================================================================
def compute_riemann_zeta_critical_line(t_max=40.0, points=1000, terms=300):
    t = np.linspace(0.1, t_max, points)
    n = np.arange(1, terms + 1)[:, None]
    alternating_signs = (-1.0) ** (n - 1)
    eta = np.sum(alternating_signs * (n ** (-0.5 - 1j * t[None, :])), axis=0)
    zeta = eta / (1.0 - 2.0 ** (0.5 - 1j * t))
    return t, np.real(zeta), np.imag(zeta), np.abs(zeta)


def compute_mandelbrot_set(x_min=-2.0, x_max=0.8, y_min=-1.2, y_max=1.2, res=350, max_iter=80):
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    C = x[None, :] + 1j * y[:, None]
    Z = np.zeros_like(C)
    escape_grid = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask] ** 2 + C[mask]
        escape_grid[mask] = i

    return x, y, escape_grid


def solve_lorenz_attractor(sigma=10.0, rho=28.0, beta=8.0 / 3.0, steps=4000, dt=0.01):
    xyz = np.zeros((steps, 3))
    xyz = (0.1, 0.0, 0.0)
    for i in range(steps - 1):
        x, y, z = xyz[i]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xyz[i + 1] = (x + dx * dt, y + dy * dt, z + dz * dt)
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]


# ==============================================================================
# 2. COMPUTING & CRYPTOGRAPHY MODULE
# ==============================================================================
def run_elementary_cellular_automaton(rule_number=30, width=121, steps=60):
    rule_bits = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=int)
    grid = np.zeros((steps, width), dtype=int)
    grid[0, width // 2] = 1

    for t in range(1, steps):
        left = np.roll(grid[t - 1], 1)
        center = grid[t - 1]
        right = np.roll(grid[t - 1], -1)
        neighborhood = (left << 2) | (center << 1) | right
        grid[t] = rule_bits[neighborhood]

    return grid


def shannon_entropy(data_string):
    if not data_string:
        return 0.0, {}
    length = len(data_string)
    counts = {}
    for ch in data_string:
        counts[ch] = counts.get(ch, 0) + 1

    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy, counts


def rsa_key_demonstration(p, q, message_int):
    n = p * q
    phi_n = (p - 1) * (q - 1)
    e = 3
    while math.gcd(e, phi_n) != 1:
        e += 2
    d = pow(e, -1, phi_n)
    m = message_int % n
    ciphertext = pow(m, e, n)
    decrypted = pow(ciphertext, d, n)
    return {
        "Modulus (n = p*q)": n,
        "Euler Totient phi(n)": phi_n,
        "Public Exponent (e)": e,
        "Private Exponent (d)": d,
        "Original Integer (m mod n)": m,
        "Encrypted Ciphertext (c)": ciphertext,
        "Decrypted Plaintext": decrypted,
    }


# ==============================================================================
# 3. PHYSICAL SCIENCE & EPHEMERIS MODULE
# ==============================================================================
SAMPLE_RATE = 44100

INNER_PLANETS = {
    "Mercury": {"period": 0.240846, "semi_major_axis": 0.387, "eccentricity": 0.2056, "color": "#a8a8a8"},
    "Venus":   {"period": 0.615198, "semi_major_axis": 0.723, "eccentricity": 0.0067, "color": "#e6c280"},
    "Earth":   {"period": 1.000000, "semi_major_axis": 1.000, "eccentricity": 0.0167, "color": "#4fa3ff"},
    "Mars":    {"period": 1.880815, "semi_major_axis": 1.524, "eccentricity": 0.0934, "color": "#ff6b4a"},
}

JPL_BODY_IDS = {"Mercury": "199", "Venus": "299", "Earth": "399", "Mars": "499"}

INTERNATIONAL_MODELS = {
    "NASA_JPL_DE440":  {"Mercury": 0.38709893, "Venus": 0.72333199, "Earth": 1.00000011, "Mars": 1.52366231},
    "EUROPE_INPOP21a": {"Mercury": 0.38709893, "Venus": 0.72333199, "Earth": 1.00000011, "Mars": 1.52366231},
    "RUSSIA_EPM2021":  {"Mercury": 0.38709893, "Venus": 0.72333199, "Earth": 1.00000011, "Mars": 1.52366231},
    "CHINA_PMO_REF":   {"Mercury": 0.38709893, "Venus": 0.72333199, "Earth": 1.00000011, "Mars": 1.52366231},
}


def solve_kepler(M, e, iterations=12):
    M_arr = np.asarray(M, dtype=float)
    E = M_arr.copy()
    for _ in range(iterations):
        E = E - (E - e * np.sin(E) - M_arr) / (1.0 - e * np.cos(E))
    return E


def compute_orbit_coordinates(a, e, M):
    E = solve_kepler(M, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1.0 - e**2) * np.sin(E)
    return x, y


def generate_chladni_plate(n, m, resolution=250):
    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(n * np.pi * X) * np.cos(m * np.pi * Y) - np.cos(m * np.pi * X) * np.cos(n * np.pi * Y)
    return X, Y, Z


def fetch_nasa_jpl_live_distance(body_id, start_date="2026-11-14", stop_date="2026-11-15"):
    params = {
        "format": "json",
        "COMMAND": f"'{body_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",
        "START_TIME": f"'{start_date}'",
        "STOP_TIME": f"'{stop_date}'",
        "STEP_SIZE": "'1 d'",
        "OUT_UNITS": "'AU-D'",
    }
    url = "https://nasa.gov?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
            for line in payload.get("result", "").splitlines():
                if "X =" in line and "Y =" in line:
                    clean_line = line.replace("X =", "").replace("Y =", "").replace("Z =", "")
                    x_str, y_str, z_str = clean_line.split()[:3]
                    x, y, z = float(x_str), float(y_str), float(z_str)
                    return float(np.sqrt(x**2 + y**2 + z**2))
    except Exception:
        return None
    return None


# ==============================================================================
# 4. QUANTUM MECHANICS MODULE
# ==============================================================================
def solve_1d_schrodinger(potential_type="Harmonic Oscillator", num_states=4, points=400):
    x = np.linspace(-5.0, 5.0, points)
    dx = float(x - x)

    if potential_type == "Harmonic Oscillator":
        V = 0.5 * (x ** 2)
    elif potential_type == "Infinite Square Well":
        V = np.zeros_like(x)
        V[np.abs(x) > 3.0] = 1e4
    elif potential_type == "Double Well (Quantum Tunneling)":
        V = 0.25 * (x ** 2 - 4.0) ** 2
    else:
        V = np.zeros_like(x)

    diag = (1.0 / (dx ** 2)) + V
    off_diag = -0.5 / (dx ** 2) * np.ones(points - 1)

    energies, wavefunctions = eigh_tridiagonal(diag, off_diag, select="i", select_range=(0, num_states - 1))
    for i in range(num_states):
        norm = np.sqrt(np.trapezoid(wavefunctions[:, i] ** 2, x))
        wavefunctions[:, i] /= norm

    return x, V, energies, wavefunctions


def qubit_bloch_state(theta_deg, phi_deg):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    alpha = np.cos(theta / 2.0)
    beta = np.exp(1j * phi) * np.sin(theta / 2.0)

    prob_0 = float(np.abs(alpha) ** 2)
    prob_1 = float(np.abs(beta) ** 2)

    bx = float(np.sin(theta) * np.cos(phi))
    by = float(np.sin(theta) * np.sin(phi))
    bz = float(np.cos(theta))
    return alpha, beta, prob_0, prob_1, (bx, by, bz)


def bell_state_entanglement(bell_type="Phi+ (|00> + |11>)/sqrt(2)"):
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if bell_type.startswith("Phi+"):
        psi = np.array([inv_sqrt2, 0.0, 0.0, inv_sqrt2])
    elif bell_type.startswith("Phi-"):
        psi = np.array([inv_sqrt2, 0.0, 0.0, -inv_sqrt2])
    elif bell_type.startswith("Psi+"):
        psi = np.array([0.0, inv_sqrt2, inv_sqrt2, 0.0])
    else:
        psi = np.array([0.0, inv_sqrt2, -inv_sqrt2, 0.0])

    psi_matrix = psi.reshape(2, 2)
    rho_A = psi_matrix @ psi_matrix.conj().T
    eigvals = np.linalg.eigvalsh(rho_A)
    eigvals = eigvals[eigvals > 1e-12]
    entanglement_entropy = float(-np.sum(eigvals * np.log2(eigvals)))

    p00, p01, p10, p11 = [float(np.abs(amp) ** 2) for amp in psi]
    probs = {
        "|00>": p00,
        "|01>": p01,
        "|10>": p10,
        "|11>": p11,
    }
    return psi, rho_A, entanglement_entropy, probs


# ==============================================================================
# 5. SYMBOLIC AI & ACADEMIC LITERATURE MODULE
# ==============================================================================
def solve_symbolic_math(expression_str, mode="Simplify / Evaluate", variable_str="x"):
    try:
        var = sp.Symbol(variable_str.strip() or "x")
        if "=" in expression_str and mode == "Solve Equation (= 0)":
            lhs_str, rhs_str = expression_str.split("=", 1)
            expr = sp.sympify(lhs_str) - sp.sympify(rhs_str)
        else:
            expr = sp.sympify(expression_str)

        if mode == "Solve Equation (= 0)":
            result = sp.solve(expr, var)
        elif mode == "Differentiate (d/dx)":
            result = sp.diff(expr, var)
        elif mode == "Integrate (∫ dx)":
            result = sp.integrate(expr, var)
        elif mode == "Series Expansion (Taylor)":
            result = sp.series(expr, var, 0, 6).removeO()
        else:
            result = sp.simplify(expr)

        return {
            "status": "success",
            "input_latex": sp.latex(expr),
            "result_str": str(result),
