import math
import numpy as np


def prime_factorization(n):
    """Returns prime factors of an integer n as a dictionary {prime: exponent}."""
    n = abs(int(n))
    if n < 2:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def fibonacci_sequence(length):
    """Generates Fibonacci numbers and consecutive Golden Ratio (Phi) approximations."""
    seq = [1, 1]
    while len(seq) < max(2, length):
        seq.append(seq[-1] + seq[-2])
    seq = seq[:length]

    rows = []
    phi_exact = (1.0 + math.sqrt(5.0)) / 2.0
    for i, val in enumerate(seq):
        ratio = (seq[i] / seq[i - 1]) if i > 0 else 1.0
        error = abs(ratio - phi_exact) if i > 0 else abs(1.0 - phi_exact)
        rows.append({
            "Index (n)": i + 1,
            "Fibonacci F(n)": val,
            "Ratio F(n)/F(n-1)": round(ratio, 10),
            "Phi Convergence Error": f"{error:.2e}",
        })
    return rows


def continued_fraction(value, max_terms=10):
    """Computes continued fraction expansion and rational convergents for any real number."""
    x = float(value)
    terms = []
    convergents = []
    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0

    for _ in range(max_terms):
        a = math.floor(x)
        terms.append(a)
        p = a * p_prev1 + p_prev2
        q = a * q_prev1 + q_prev2
        convergents.append({
            "Term (a_n)": a,
            "Rational Approximation": f"{p}/{q}",
            "Decimal Value": round(p / q, 10),
            "Residual Error": f"{abs(value - (p / q)):.2e}",
        })
        p_prev2, p_prev1 = p_prev1, p
        q_prev2, q_prev1 = q_prev1, q

        frac = x - a
        if frac < 1e-12:
            break
        x = 1.0 / frac

    return terms, convergents


def modular_multiplication_table(modulus, multiplier):
    """Generates modular arithmetic chord coordinates (Vortex/Cardioid geometry)."""
    indices = np.arange(modulus)
    targets = (indices * multiplier) % modulus
    angles_start = 2.0 * np.pi * indices / modulus
    angles_end = 2.0 * np.pi * targets / modulus

    chords = []
    for i in range(modulus):
        chords.append((
            np.cos(angles_start[i]), np.sin(angles_start[i]),
            np.cos(angles_end[i]), np.sin(angles_end[i]),
        ))
    return chords
