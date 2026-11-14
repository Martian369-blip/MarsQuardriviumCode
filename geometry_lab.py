import numpy as np


def generate_polygon_vertices(sides, radius=1.0, rotation_deg=0.0):
    """Generates exact 2D Euclidean coordinates for a regular n-gon."""
    rot = np.deg2rad(rotation_deg)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + rot
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y


def generate_flower_of_life(rings=2, radius=1.0):
    """Generates circle center coordinates for a hexagonal Flower of Life lattice."""
    centers = [(0.0, 0.0)]
    if rings >= 1:
        for i in range(6):
            angle = i * np.pi / 3.0
            centers.append((radius * np.cos(angle), radius * np.sin(angle)))
    if rings >= 2:
        for i in range(6):
            angle = i * np.pi / 3.0
            centers.append((2.0 * radius * np.cos(angle), 2.0 * radius * np.sin(angle)))
            mid_angle = angle + np.pi / 6.0
            dist = radius * np.sqrt(3.0)
            centers.append((dist * np.cos(mid_angle), dist * np.sin(mid_angle)))
    return centers


def generate_golden_spiral(turns=4, points=1000):
    """Generates a logarithmic spiral governed by the Golden Ratio Phi."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    b = np.log(phi) / (np.pi / 2.0)
    theta = np.linspace(0, turns * 2.0 * np.pi, points)
    r = np.exp(b * theta)
    r = r / np.max(r)  # Normalize outer radius to 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def generate_synodic_rose(period_1, period_2, orbits=8, points=1500):
    """Draws geometric conjunction lines between two orbiting bodies (e.g., Earth & Venus)."""
    t = np.linspace(0, orbits * max(period_1, period_2), points)
    theta1 = 2.0 * np.pi * t / period_1
    theta2 = 2.0 * np.pi * t / period_2

    r1, r2 = 1.0, 0.723
    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)
    return x1, y1, x2, y2
