import io
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from arithmetic_lab import (
    prime_factorization,
    fibonacci_sequence,
    continued_fraction,
    modular_multiplication_table,
)
from geometry_lab import (
    generate_polygon_vertices,
    generate_flower_of_life,
    generate_golden_spiral,
    generate_synodic_rose,
)
from harmonics_lab import (
    TUNING_SYSTEMS,
    compute_fft_spectrum,
    generate_chladni_plate,
)
from quadrivium_engine import (
    INNER_PLANETS,
    OUTER_PLANETS,
    SAMPLE_RATE,
    compute_orbit_coordinates,
)
from benchmark_ephemeris import (
    JPL_BODY_IDS,
    INTERNATIONAL_MODELS,
    fetch_nasa_jpl_live_distance,
)

st.set_page_config(page_title="Quadrivium Research Engine", layout="wide")
st.title("Quadrivium Research Engine")
st.markdown("Computational Research Workbench across **Arithmetic, Geometry, Harmonics, and Astronomy**.")

tab1, tab2, tab3, tab4 = st.tabs([
    "I. Arithmetic Lab",
    "II. Geometry Lab",
    "III. Harmonics & Cymatics Lab",
    "IV. Astronomy & Ephemeris Lab",
])

# ==============================================================================
# TAB I: ARITHMETIC LAB
# ==============================================================================
with tab1:
    st.header("Pillar I: Number Theory, Continued Fractions & Modular Geometry")
    col_a1, col_a2 = st.columns([1, 1.2])

    with col_a1:
        st.subheader("1. Prime Factorization & Continued Fractions")
        target_int = st.number_input("Integer for Prime Decomposition", min_value=2, max_value=10**12, value=432, step=1)
        factors = prime_factorization(target_int)
        factor_str = " × ".join([f"{p}^{e}" if e > 1 else f"{p}" for p, e in factors.items()])
        st.info(f"**Prime Decomposition of {target_int}:** `{factor_str}`")

        real_val = st.number_input("Real Constant for Rational Approximation (e.g., Phi=1.6180339887, Pi=3.14159265)", value=1.61803398875, format="%.11f")
        terms, convergents = continued_fraction(real_val, max_terms=8)
        st.write(f"**Continued Fraction Terms:** `{terms}`")
        st.dataframe(pd.DataFrame(convergents), use_container_width=True, hide_index=True)

        st.subheader("2. Fibonacci & Golden Ratio (φ) Convergence")
        fib_len = st.slider("Sequence Depth", min_value=5, max_value=30, value=12)
        fib_df = pd.DataFrame(fibonacci_sequence(fib_len))
        st.dataframe(fib_df, use_container_width=True, hide_index=True)
        st.download_button("Export Arithmetic Data (CSV)", fib_df.to_csv(index=False), "arithmetic_phi_convergence.csv", "text/csv")

    with col_a2:
        st.subheader("3. Modular Arithmetic Lattice (Vortex Chord Geometry)")
        mod_n = st.slider("Modulus (Points on Circle)", min_value=12, max_value=360, value=144, step=1)
        mult_k = st.slider("Multiplier (k)", min_value=2, max_value=99, value=2, step=1)

        chords = modular_multiplication_table(mod_n, mult_k)
        fig_mod, ax_mod = plt.subplots(figsize=(6.5, 6.5), facecolor="#0d1117")
        ax_mod.set_facecolor("#0d1117")
        for x1, y1, x2, y2 in chords:
            ax_mod.plot([x1, x2], [y1, y2], color="#38bdf8", alpha=0.45, linewidth=0.8)
        ax_mod.set_title(f"Modular Chord Mapping: (n × {mult_k}) mod {mod_n}", color="white")
        ax_mod.axis("equal")
        ax_mod.axis("off")
        st.pyplot(fig_mod)

# ==============================================================================
# TAB II: GEOMETRY LAB
# ==============================================================================
with tab2:
    st.header("Pillar II: Euclidean Symmetry, Sacred Lattices & Synodic Geometry")
    col_g1, col_g2 = st.columns([1, 1.3])

    with col_g1:
        geom_mode = st.selectbox("Select Geometric System", [
            "Regular Polygon & Dihedral Symmetry",
            "Flower of Life Hexagonal Lattice",
            "Logarithmic Golden Spiral",
            "Synodic Planetary Rose (Earth–Venus 8:13 Resonance)",
        ])

        fig_geo, ax_geo = plt.subplots(figsize=(6.5, 6.5), facecolor="#0d1117")
        ax_geo.set_facecolor("#0d1117")

        if geom_mode == "Regular Polygon & Dihedral Symmetry":
            sides = st.slider("Number of Sides (n)", min_value=3, max_value=24, value=6)
            gx, gy = generate_polygon_vertices(sides)
            # Connect all vertices (complete graph K_n)
            for i in range(sides):
                for j in range(i + 1, sides):
                    ax_geo.plot([gx[i], gx[j]], [gy[i], gy[j]], color="#f59e0b", alpha=0.5, linewidth=1.0)
            ax_geo.scatter(gx, gy, color="#ffffff", s=40, zorder=5)
            coord_df = pd.DataFrame({"Vertex": np.arange(1, sides + 1), "X": np.round(gx, 6), "Y": np.round(gy, 6)})
            st.dataframe(coord_df, use_container_width=True, hide_index=True)
            st.download_button("Export Vertex Coordinates (CSV)", coord_df.to_csv(index=False), f"polygon_{sides}_vertices.csv", "text/csv")

        elif geom_mode == "Flower of Life Hexagonal Lattice":
            rings = st.slider("Concentric Lattice Rings", min_value=1, max_value=2, value=2)
            centers = generate_flower_of_life(rings=rings, radius=1.0)
            for cx, cy in centers:
                circle = plt.Circle((cx, cy), 1.0, color="#34d399", fill=False, linewidth=1.2, alpha=0.7)
                ax_geo.add_patch(circle)
            ax_geo.set_xlim(-3.2, 3.2)
            ax_geo.set_ylim(-3.2, 3.2)
            st.write(f"**Total Overlapping Circles:** `{len(centers)}`")

        elif geom_mode == "Logarithmic Golden Spiral":
            turns = st.slider("Spiral Turns", min_value=2, max_value=8, value=5)
            sx, sy = generate_golden_spiral(turns=turns)
            ax_geo.plot(sx, sy, color="#fbbf24", linewidth=1.8)

        elif geom_mode == "Synodic Planetary Rose (Earth–Venus 8:13 Resonance)":
            orbits = st.slider("Earth Years Elapsed", min_value=1, max_value=16, value=8)
            x1, y1, x2, y2 = generate_synodic_rose(1.0, 0.615198, orbits=orbits, points=320)
            for i in range(len(x1)):
                ax_geo.plot([x1[i], x2[i]], [y1[i], y2[i]], color="#a78bfa", alpha=0.28, linewidth=0.7)
            ax_geo.plot(x1, y1, color="#4fa3ff", linewidth=1.2, label="Earth (1.00 AU)")
            ax_geo.plot(x2, y2, color="#e6c280", linewidth=1.2, label="Venus (0.723 AU)")
            ax_geo.legend(facecolor="#161b22", edgecolor="none", labelcolor="white")

        ax_geo.axis("equal")
        ax_geo.axis("off")

    with col_g2:
        st.pyplot(fig_geo)

# ==============================================================================
# TAB III: HARMONICS & CYMATICS LAB
# ==============================================================================
with tab3:
    st.header("Pillar III: Acoustic Tuning Systems, FFT Spectrum & Chladni Cymatics")
    col_h1, col_h2 = st.columns([1, 1.2])

    with col_h1:
        st.subheader("1. Tuning System Synthesizer & FFT Analyzer")
        tuning_choice = st.selectbox("Select Tuning System", list(TUNING_SYSTEMS.keys()))
        root_hz = st.number_input("Root Frequency (Hz)", min_value=20.0, max_value=1000.0, value=256.0, step=1.0)
        selected_intervals = st.multiselect(
            "Select Harmonic Intervals to Combine",
            list(TUNING_SYSTEMS[tuning_choice].keys()),
            default=["Unison (1:1)", "Perfect Fifth (3:2)", "Octave (2:1)"] if "Unison (1:1)" in TUNING_SYSTEMS[tuning_choice] else ["Unison", "Perfect Fifth", "Octave"],
        )

        dur_h = 4.0
        t_h = np.linspace(0, dur_h, int(SAMPLE_RATE * dur_h), endpoint=False)
        chord_wave = np.zeros_like(t_h)
        interval_table = []

        for label in selected_intervals:
            ratio = TUNING_SYSTEMS[tuning_choice][label]
            freq_val = root_hz * ratio
            chord_wave += np.sin(2.0 * np.pi * freq_val * t_h)
            interval_table.append({"Interval": label, "Ratio": round(ratio, 6), "Frequency (Hz)": round(freq_val, 3)})

        if len(selected_intervals) > 0:
            norm_wave = np.int16((chord_wave / np.max(np.abs(chord_wave))) * 32767)
            buf_h = io.BytesIO()
            write(buf_h, SAMPLE_RATE, norm_wave)
            buf_h.seek(0)
            st.audio(buf_h, format="audio/wav")
            st.dataframe(pd.DataFrame(interval_table), use_container_width=True, hide_index=True)

            # Plot FFT Spectrum
            xf, mag = compute_fft_spectrum(chord_wave[:SAMPLE_RATE], SAMPLE_RATE)
            fig_fft, ax_fft = plt.subplots(figsize=(6, 2.8), facecolor="#0d1117")
            ax_fft.set_facecolor("#0d1117")
            ax_fft.plot(xf, mag, color="#38bdf8", linewidth=1.2)
            ax_fft.set_xlim(0, root_hz * 2.5)
            ax_fft.set_title("Fast Fourier Transform (FFT) Frequency Spectrum", color="white", fontsize=10)
            ax_fft.set_xlabel("Frequency (Hz)", color="gray")
            ax_fft.tick_params(colors="gray")
            st.pyplot(fig_fft)

    with col_h2:
        st.subheader("2. 2D Chladni Cymatic Nodal Wave Patterns")
        mode_n = st.slider("Modal Number (n)", min_value=1, max_value=12, value=3)
        mode_m = st.slider("Modal Number (m)", min_value=1, max_value=12, value=5)

        X_c, Y_c, Z_c = generate_chladni_plate(mode_n, mode_m)
        fig_cym, ax_cym = plt.subplots(figsize=(6, 6), facecolor="#0d1117")
        ax_cym.set_facecolor("#0d1117")
        ax_cym.contour(X_c, Y_c, Z_c, levels=[0], colors="#fbbf24", linewidths=1.8)
        ax_cym.set_title(f"Chladni Nodal Geometry (Mode n={mode_n}, m={mode_m})", color="white")
        ax_cym.axis("equal")
        ax_cym.axis("off")
        st.pyplot(fig_cym)

# ==============================================================================
# TAB IV: ASTRONOMY & EPHEMERIS BENCHMARK LAB
