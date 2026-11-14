import io
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from math_lab import (
    compute_riemann_zeta_critical_line,
    compute_mandelbrot_set,
    solve_lorenz_attractor,
)
from computing_lab import (
    run_elementary_cellular_automaton,
    shannon_entropy,
    rsa_key_demonstration,
)
from science_lab import (
    SAMPLE_RATE,
    INNER_PLANETS,
    JPL_BODY_IDS,
    INTERNATIONAL_MODELS,
    compute_orbit_coordinates,
    generate_chladni_plate,
    fetch_nasa_jpl_live_distance,
)
from quantum_lab import (
    solve_1d_schrodinger,
    qubit_bloch_state,
    bell_state_entanglement,
)
from research_ai_lab import (
    solve_symbolic_math,
    query_arxiv_papers,
    query_crossref_research,
    query_encyclopedic_synthesis,
)

st.set_page_config(page_title="Unified Research Engine", layout="wide")
st.title("Unified Computational Research Engine")
st.markdown("Research Workbench across **Mathematics, Computing, Physical Science, Quantum Mechanics, and Multi-Domain Literature Synthesis**.")

tab_math, tab_comp, tab_sci, tab_qm, tab_ai = st.tabs([
    "I. Mathematics Lab",
    "II. Computing & Cryptography Lab",
    "III. Physical Science & Ephemeris Lab",
    "IV. Quantum Mechanics Lab",
    "V. AI Research Assistant",
])

# ==============================================================================
# TAB I: MATHEMATICS LAB
# ==============================================================================
with tab_math:
    st.header("Mathematics: Complex Analysis, Fractals & Nonlinear Dynamics")
    math_domain = st.selectbox("Select Mathematical Research Module", [
        "Riemann Zeta Function on the Critical Line Re(s) = 1/2",
        "Mandelbrot Complex Dynamical System",
        "Lorenz Strange Attractor (Chaos Theory)",
    ])

    if math_domain == "Riemann Zeta Function on the Critical Line Re(s) = 1/2":
        t_max = st.slider("Imaginary Axis Upper Bound (t)", min_value=15.0, max_value=80.0, value=40.0)
        t, re_z, im_z, abs_z = compute_riemann_zeta_critical_line(t_max=t_max)

        fig_z, ax_z = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
        ax_z.set_facecolor("#0d1117")
        ax_z.plot(t, abs_z, color="#38bdf8", linewidth=1.6, label="|ζ(0.5 + it)| Magnitude")
        ax_z.plot(t, re_z, color="#34d399", alpha=0.6, linewidth=1.0, label="Re(ζ)")
        ax_z.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_z.set_title("Riemann Zeta Function along Critical Line s = 0.5 + it", color="white")
        ax_z.set_xlabel("Imaginary Component (t)", color="gray")
        ax_z.tick_params(colors="gray")
        ax_z.legend(facecolor="#161b22", edgecolor="none", labelcolor="white")
        st.pyplot(fig_z)

        zeta_df = pd.DataFrame({"t": np.round(t, 4), "Re(zeta)": np.round(re_z, 6), "Im(zeta)": np.round(im_z, 6), "|zeta|": np.round(abs_z, 6)})
        st.download_button("Export Zeta Critical Line Data (CSV)", zeta_df.to_csv(index=False), "riemann_zeta_critical_line.csv", "text/csv")

    elif math_domain == "Mandelbrot Complex Dynamical System":
        max_iter = st.slider("Escape-Time Iterations", min_value=30, max_value=200, value=80)
        x, y, grid = compute_mandelbrot_set(max_iter=max_iter)

        fig_m, ax_m = plt.subplots(figsize=(7, 6), facecolor="#0d1117")
        ax_m.imshow(grid, extent=[x.min(), x.max(), y.min(), y.max()], cmap="inferno", origin="lower")
        ax_m.set_title("Mandelbrot Set Stability Boundary (z_{n+1} = z_n^2 + c)", color="white")
        ax_m.set_xlabel("Re(c)", color="gray")
        ax_m.set_ylabel("Im(c)", color="gray")
        ax_m.tick_params(colors="gray")
        st.pyplot(fig_m)

    elif math_domain == "Lorenz Strange Attractor (Chaos Theory)":
        rho = st.slider("Rayleigh Number (ρ)", min_value=10.0, max_value=45.0, value=28.0)
        lx, ly, lz = solve_lorenz_attractor(rho=rho)

        fig_l, ax_l = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
        ax_l.set_facecolor("#0d1117")
        ax_l.plot(lx, lz, color="#fbbf24", linewidth=0.7, alpha=0.85)
        ax_l.set_title(f"Lorenz Phase-Space Projection (X–Z Plane, ρ={rho})", color="white")
        ax_l.set_xlabel("X (Convective Intensity)", color="gray")
        ax_l.set_ylabel("Z (Temperature Gradient)", color="gray")
        ax_l.tick_params(colors="gray")
        st.pyplot(fig_l)

# ==============================================================================
# TAB II: COMPUTING & CRYPTOGRAPHY LAB
# ==============================================================================
with tab_comp:
    st.header("Computing: Cellular Automata, Information Theory & Cryptography")
    col_c1, col_c2 = st.columns([1.2, 1])

    with col_c1:
        st.subheader("1. Elementary Cellular Automata (Emergent Computation)")
        rule_num = st.selectbox("Select Wolfram Rule", [30, 90, 110, 184], index=2)
        steps = st.slider("Evolution Generations", min_value=30, max_value=150, value=75)
        ca_grid = run_elementary_cellular_automaton(rule_number=rule_num, width=steps * 2 + 1, steps=steps)

        fig_ca, ax_ca = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
        ax_ca.imshow(ca_grid, cmap="binary", interpolation="nearest")
        ax_ca.set_title(f"1D Cellular Automaton — Rule {rule_num}", color="white")
        ax_ca.axis("off")
        st.pyplot(fig_ca)

    with col_c2:
        st.subheader("2. Shannon Information Entropy")
        sample_text = st.text_input("Input Symbol Sequence", value="QUADRIVIUM_QUANTUM_COMPUTING_2026")
        h_val, counts = shannon_entropy(sample_text)
        st.metric("Shannon Entropy H(X)", f"{h_val:.4f} bits / symbol")

        st.subheader("3. RSA Modular Cryptography Engine")
        p_prime = st.selectbox("Prime p", [61, 67, 71, 73, 79, 83], index=0)
        q_prime = st.selectbox("Prime q", [53, 59, 89, 97, 101, 103], index=0)
        msg_int = st.number_input("Integer Message Payload", min_value=2, max_value=3000, value=369)
        rsa_res = rsa_key_demonstration(p_prime, q_prime, msg_int)
        st.dataframe(pd.DataFrame([rsa_res]), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB III: PHYSICAL SCIENCE & EPHEMERIS LAB
# ==============================================================================
with tab_sci:
    st.header("Physical Science: Orbital Mechanics, Wave Physics & Ephemeris Benchmarks")
    col_s1, col_s2 = st.columns([1.1, 1])

    with col_s1:
        st.subheader("1. Keplerian Orbital Mechanics & Sonification")
        base_freq = st.slider("Earth Orbital Reference Frequency (Hz)", min_value=55.0, max_value=440.0, value=110.0)
        fig_orb, ax_orb = plt.subplots(figsize=(6, 6), facecolor="#0d1117")
        ax_orb.set_facecolor("#0d1117")
        origin = np.zeros(1)
        ax_orb.scatter(origin, origin, color="#ffcc00", s=160, label="Sun (Origin)", zorder=5)

        M_full = np.linspace(0, 2 * np.pi, 800)
        t_audio = np.linspace(0, 5.0, int(SAMPLE_RATE * 5.0), endpoint=False)
        wave_sum = np.zeros_like(t_audio)

        for name, data in INNER_PLANETS.items():
            freq = base_freq / data["period"]
            x, y = compute_orbit_coordinates(data["semi_major_axis"], data["eccentricity"], M_full)
            ax_orb.plot(x, y, color=data["color"], linewidth=1.5, label=f"{name} ({freq:.2f} Hz)")
            wave_sum += np.sin(2 * np.pi * freq * t_audio)

        ax_orb.axis("equal")
        ax_orb.tick_params(colors="gray")
        ax_orb.legend(facecolor="#161b22", edgecolor="none", labelcolor="white")
        st.pyplot(fig_orb)

        norm_audio = np.int16((wave_sum / np.max(np.abs(wave_sum))) * 32767)
        buf = io.BytesIO()
        write(buf, SAMPLE_RATE, norm_audio)
        buf.seek(0)
        st.audio(buf, format="audio/wav")

    with col_s2:
        st.subheader("2. 2D Acoustic Wave Nodal Geometry (Chladni Plate)")
        n_mode = st.slider("Wave Mode n", min_value=1, max_value=10, value=3)
        m_mode = st.slider("Wave Mode m", min_value=1, max_value=10, value=5)
        Xc, Yc, Zc = generate_chladni_plate(n_mode, m_mode)

        fig_ch, ax_ch = plt.subplots(figsize=(5, 5), facecolor="#0d1117")
        ax_ch.set_facecolor("#0d1117")
        ax_ch.contour(Xc, Yc, Zc, levels=[0.0], colors="#38bdf8", linewidths=1.6)
        ax_ch.axis("equal")
        ax_ch.axis("off")
        st.pyplot(fig_ch)

    st.divider()
    st.subheader("3. International Space Agency Ephemeris Benchmark (NASA JPL, ESA, Russia, China)")
    if st.button("Run Multi-Agency Orbital Comparison"):
        AU_TO_KM = 149597870.7
        rows = []
        for name, data in INNER_PLANETS.items():
            qx, qy = compute_orbit_coordinates(data["semi_major_axis"], data["eccentricity"], M=0.0)
            q_rad = float(np.sqrt(qx**2 + qy**2))
            live_au = fetch_nasa_jpl_live_distance(JPL_BODY_IDS[name])
            ref_au = live_au if live_au is not None else INTERNATIONAL_MODELS["NASA_JPL_DE440"][name]
            rows.append({
                "Planet": name,
                "Engine Perihelion (AU)": round(q_rad, 6),
                "NASA JPL Live (AU)": round(live_au, 6) if live_au else "DE440 Ref",
                "Spatial Delta (km)": f"{abs(q_rad - ref_au) * AU_TO_KM:,.1f}",
                "EPM2021 / INPOP21a / PMO Ref (AU)": INTERNATIONAL_MODELS["RUSSIA_EPM2021"][name],
            })
        bench_df = pd.DataFrame(rows)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.download_button("Export Ephemeris Benchmark (CSV)", bench_df.to_csv(index=False), "ephemeris_benchmark.csv", "text/csv")

# ==============================================================================
# TAB IV: QUANTUM MECHANICS LAB
# ==============================================================================
with tab_qm:
