import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

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

st.set_page_config(page_title="Quadrivium Engine", layout="wide")
st.title("Quadrivium Engine: Orbital Harmonics & Keplerian Geometry")
st.markdown("Unifying **Arithmetic, Geometry, Harmonics, and Astronomy** with independent international ephemeris benchmarking.")

# Sidebar Controls
st.sidebar.header("Simulation Controls")
base_freq = st.sidebar.slider("Earth Fundamental Frequency (Hz)", min_value=55.0, max_value=440.0, value=110.0, step=1.0)
duration = st.sidebar.slider("Audio Duration (Seconds)", min_value=3, max_value=20, value=8)
include_outer = st.sidebar.checkbox("Include Outer Planets (Jupiter & Saturn)", value=False)

active_planets = dict(INNER_PLANETS)
if include_outer:
    active_planets.update(OUTER_PLANETS)

col1, col2 = st.columns([1.2, 1])

# Left Column: Geometry Plot
with col1:
    st.subheader("Keplerian Orbital Geometry")
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    origin = np.zeros(1)
    ax.scatter(origin, origin, color="#ffcc00", s=180, label="Sun (Origin)", zorder=5)

    M_full = np.linspace(0, 2 * np.pi, 1000)
    for name, data in active_planets.items():
        freq = base_freq / data["period"]
        x, y = compute_orbit_coordinates(data["semi_major_axis"], data["eccentricity"], M_full)
        ax.plot(x, y, color=data["color"], linewidth=1.6, label=f"{name} ({freq:.2f} Hz)")

    ax.set_xlabel("Astronomical Units (AU)", color="gray")
    ax.set_ylabel("Astronomical Units (AU)", color="gray")
    ax.tick_params(colors="gray")
    ax.grid(True, linestyle="--", alpha=0.15)
    ax.axis("equal")
    ax.legend(facecolor="#161b22", edgecolor="none", labelcolor="white", loc="upper right")
    st.pyplot(fig)

# Right Column: Harmonics Audio Player & Frequency Table
with col2:
    st.subheader("Harmonics: Music of the Spheres")
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    master_wave = np.zeros_like(t)

    freq_rows = []
    for name, data in active_planets.items():
        freq = base_freq / data["period"]
        speed_mod = 1.0 + data["eccentricity"] * np.sin(2 * np.pi * (1.0 / data["period"]) * t)
        master_wave += np.sin(2 * np.pi * freq * t * speed_mod)
        freq_rows.append({"Planet": name, "Orbital Period (yr)": data["period"], "Frequency (Hz)": round(freq, 2)})

    master_wave = np.int16((master_wave / np.max(np.abs(master_wave))) * 32767)
    wav_buffer = io.BytesIO()
    write(wav_buffer, SAMPLE_RATE, master_wave)
    wav_buffer.seek(0)

    st.audio(wav_buffer, format="audio/wav")
    st.dataframe(freq_rows, use_container_width=True, hide_index=True)

# Bottom Section: Independent International Benchmark
st.divider()
st.subheader("Independent Benchmark: Quadrivium Engine vs. International Ephemerides")
if st.button("Run International Ephemeris Comparison"):
    AU_TO_KM = 149597870.7
    bench_rows = []
    for name, data in INNER_PLANETS.items():
        qx, qy = compute_orbit_coordinates(data["semi_major_axis"], data["eccentricity"], M=0.0)
        q_radius = np.sqrt(qx**2 + qy**2)
        q_freq = base_freq / data["period"]

        live_au = fetch_nasa_jpl_live_distance(JPL_BODY_IDS[name])
        ref_au = live_au if live_au is not None else INTERNATIONAL_MODELS["NASA_JPL_DE440"][name]
        delta_km = abs(q_radius - ref_au) * AU_TO_KM

        intl_freq = base_freq / (INTERNATIONAL_MODELS["RUSSIA_EPM2021"][name] ** 1.5)
        bench_rows.append({
            "Planet": name,
            "Quadrivium Perihelion (AU)": round(float(q_radius), 6),
            "NASA JPL Live (AU)": round(float(live_au), 6) if live_au else "Offline (DE440 Ref)",
            "Spatial Delta (km)": f"{delta_km:,.1f}",
            "Freq Drift vs EPM2021/INPOP21a (Hz)": round(abs(q_freq - intl_freq), 6),
        })
    st.dataframe(bench_rows, use_container_width=True, hide_index=True)
