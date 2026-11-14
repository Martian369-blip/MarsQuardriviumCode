# Quadrivium Engine: Orbital Harmonics & Keplerian Geometry

A computational physics and digital signal processing (DSP) engine unifying the four classical pillars of the **Quadrivium**—**Arithmetic, Geometry, Harmonics, and Astronomy**—into a single Python simulation.

## Mathematical Foundation

1. **Arithmetic (Pure Number):** Derives acoustic frequencies from inverse planetary orbital periods ($f = f_0 / T$).
2. **Astronomy (Number in Space & Time):** Solves Kepler's transcendental equation ($M = E - e \sin E$) via Newton-Raphson numerical iteration.
3. **Geometry (Number in Space):** Projects eccentric anomalies onto 2D Euclidean space with the Sun positioned at the focal origin $(0, 0)$:
   $$x = a(\cos E - e), \quad y = a\sqrt{1 - e^2}\sin E$$
4. **Harmonics (Number in Time):** Synthesizes 16-bit PCM sine waves where orbital eccentricity modulates pitch frequency to reflect gravitational acceleration between perihelion and aphelion.

## Installation

```bash
git clone https://github.com
cd quadrivium-engine
pip install -r requirements.txt
```

## Usage

Run the default simulation (110 Hz Earth root, static plot + WAV audio):
```bash
python quadrivium_engine.py
```

Generate a **real-time orbital animation GIF** at **432 Hz** including **Jupiter and Saturn**:
```bash
python quadrivium_engine.py --base-freq 432 --outer --animate
```

### CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--base-freq` | Fundamental frequency for Earth in Hz | `110.0` |
| `--duration` | Length of generated audio in seconds | `10.0` |
| `--outer` | Include Jupiter and Saturn | `False` |
| `--animate` | Export animated planetary motion as `orbital_motion.gif` | `False` |

## Running Tests

Verify numerical accuracy and audio synthesis with `pytest`:
```bash
pytest -v
```
