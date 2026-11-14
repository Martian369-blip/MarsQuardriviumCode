import argparse
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ==========================================
# 1. ARITHMETIC: Astronomical Constants
# ==========================================
SAMPLE_RATE = 44100  # Hz

INNER_PLANETS = {
    "Mercury": {"period": 0.240846, "semi_major_axis": 0.387, "eccentricity": 0.2056, "color": "#a8a8a8"},
    "Venus":   {"period": 0.615198, "semi_major_axis": 0.723, "eccentricity": 0.0067, "color": "#e6c280"},
    "Earth":   {"period": 1.000000, "semi_major_axis": 1.000, "eccentricity": 0.0167, "color": "#4fa3ff"},
    "Mars":    {"period": 1.880815, "semi_major_axis": 1.524, "eccentricity": 0.0934, "color": "#ff6b4a"},
}

OUTER_PLANETS = {
    "Jupiter": {"period": 11.862,   "semi_major_axis": 5.203, "eccentricity": 0.0489, "color": "#d9a066"},
    "Saturn":  {"period": 29.457,   "semi_major_axis": 9.537, "eccentricity": 0.0565, "color": "#ead69c"},
}


# ==========================================
# 2. ASTRONOMY: Kepler's Equation Solver
# ==========================================
def solve_kepler(M, e, iterations=12):
    """Solves M = E - e * sin(E) using Newton-Raphson iteration."""
    M_arr = np.asarray(M, dtype=float)
    E = M_arr.copy()
    for _ in range(iterations):
        E = E - (E - e * np.sin(E) - M_arr) / (1.0 - e * np.cos(E))
    return E


# ==========================================
# 3. GEOMETRY: Cartesian Orbital Projection
# ==========================================
def compute_orbit_coordinates(a, e, M):
    """Maps mean anomaly M onto 2D Euclidean space (x, y) in AU."""
    E = solve_kepler(M, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1.0 - e**2) * np.sin(E)
    return x, y


# ==========================================
# 4. HARMONICS: Acoustic Wave Synthesis
# ==========================================
def generate_harmonic_audio(planets, base_freq, duration, output_wav):
    """Synthesizes multi-planetary sine waves modulated by orbital eccentricity."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    master_wave = np.zeros_like(t)

    print(f"\n--- Quadrivium Harmonic Synthesis (Root: {base_freq} Hz) ---")
    for name, data in planets.items():
        freq = base_freq / data["period"]
        speed_mod = 1.0 + data["eccentricity"] * np.sin(2 * np.pi * (1.0 / data["period"]) * t)
        wave = np.sin(2 * np.pi * freq * t * speed_mod)
        master_wave += wave
        print(f" {name:<8} | Period: {data['period']:>7.3f} yr | Frequency: {freq:>7.2f} Hz")

    max_val = np.max(np.abs(master_wave))
    if max_val > 0:
        master_wave = np.int16((master_wave / max_val) * 32767)
    write(output_wav, SAMPLE_RATE, master_wave)
    print(f"Saved audio: {output_wav}")


# ==========================================
# 5. VISUALIZATION: Static Plot & Animation
# ==========================================
def render_visuals(planets, base_freq, animate=False, output_img="orbital_geometry.png", output_gif="orbital_motion.gif"):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    origin = np.zeros(1)
    ax.scatter(origin, origin, color="#ffcc00", s=200, label="Sun (Origin)", zorder=5)

    M_full = np.linspace(0, 2 * np.pi, 1000)
    planet_markers = {}

    for name, data in planets.items():
        freq = base_freq / data["period"]
        x, y = compute_orbit_coordinates(data["semi_major_axis"], data["eccentricity"], M_full)
        ax.plot(x, y, color=data["color"], alpha=0.5, linewidth=1.2, label=f"{name} ({freq:.2f} Hz)")
        if animate:
            marker, = ax.plot([], [], marker="o", markersize=8, color=data["color"], zorder=6)
            planet_markers[name] = (marker, data)

    ax.set_title("Quadrivium: Keplerian Geometry & Harmonics", color="white", fontsize=14)
    ax.set_xlabel("Astronomical Units (AU)", color="gray")
    ax.set_ylabel("Astronomical Units (AU)", color="gray")
    ax.tick_params(colors="gray")
    ax.grid(True, linestyle="--", alpha=0.15)
    ax.axis("equal")
    ax.legend(facecolor="#161b22", edgecolor="none", labelcolor="white", loc="upper right")
    plt.tight_layout()

    if not animate:
        plt.savefig(output_img, dpi=300)
        print(f"Saved static geometry: {output_img}")
    else:
        frames = 120
        def update(frame):
            earth_years = (frame / frames) * 2.0
            for name, (marker, data) in planet_markers.items():
                M = (2 * np.pi * earth_years) / data["period"]
                px, py = compute_orbit_coordinates(data["semi_major_axis"], data["eccentricity"], M)
                marker.set_data([px], [py])
            return [m for m, _ in planet_markers.values()]

        anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
        anim.save(output_gif, writer=PillowWriter(fps=25))
        print(f"Saved real-time animation: {output_gif}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quadrivium Engine: Arithmetic, Geometry, Harmonics & Astronomy")
    parser.add_argument("--base-freq", type=float, default=110.0, help="Fundamental frequency for Earth in Hz (default: 110.0)")
    parser.add_argument("--duration", type=float, default=10.0, help="Audio duration in seconds (default: 10.0)")
    parser.add_argument("--outer", action="store_true", help="Include Jupiter and Saturn in the simulation")
    parser.add_argument("--animate", action="store_true", help="Render an animated GIF of planetary motion")
    args = parser.parse_args()

    active_planets = dict(INNER_PLANETS)
    if args.outer:
        active_planets.update(OUTER_PLANETS)

    generate_harmonic_audio(active_planets, args.base_freq, args.duration, "music_of_the_spheres.wav")
    render_visuals(active_planets, args.base_freq, animate=args.animate)


if __name__ == "__main__":
    main()
