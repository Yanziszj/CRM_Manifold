import numpy as np
import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.proximal_gradient import ProximalGradient

# Initialize the Poincaré Ball model of dimension 2
manifold = PoincareBall(dim=2)
metric = manifold.metric

# Define projections and reflections
def project_to_set(x, set_center, set_radius):
    """Projection onto a geodesic ball."""
    d = metric.dist(x, set_center)
    if d <= set_radius:
        return x
    direction = metric.log(point=x, base_point=set_center)
    scaled_direction = direction * (set_radius / d)
    return metric.exp(base_point=set_center, tangent_vec=scaled_direction)

def reflection(x, set_center, set_radius):
    """Reflection across a geodesic ball."""
    proj = project_to_set(x, set_center, set_radius)
    tangent_vec = -metric.log(base_point=proj, point=x)
    return metric.exp(base_point=proj, tangent_vec=tangent_vec)

# Douglas-Rachford Method
def douglas_rachford(x0, setA, setB, n_iter=50):
    x = x0
    history = [x0]
    for _ in range(n_iter):
        ra = reflection(x, setA['center'], setA['radius'])
        rb = reflection(ra, setB['center'], setB['radius'])
        x = manifold.midpoint(x, rb)
        history.append(x)
    return np.array(history)

# CRM with Tangent-Space Quasi-Circumcenter
def tangent_space_crm(x0, setA, setB, n_iter=50):
    x = x0
    history = [x0]
    for _ in range(n_iter):
        ra = reflection(x, setA['center'], setA['radius'])
        rb = reflection(ra, setB['center'], setB['radius'])
        # Map points to tangent space at x
        v1 = metric.log(base_point=x, point=ra)
        v2 = metric.log(base_point=x, point=rb)
        # Compute circumcenter in tangent space
        circumcenter_tangent = (v1 + v2) / 2
        # Map back to the manifold
        x = metric.exp(base_point=x, tangent_vec=circumcenter_tangent)
        history.append(x)
    return np.array(history)

# Initialize sets A and B as geodesic balls
setA = {'center': gs.array([0.2, 0.2]), 'radius': 0.3}
setB = {'center': gs.array([-0.2, -0.2]), 'radius': 0.3}

# Initial point
x0 = gs.array([0.0, 0.5])

# Run the algorithms
n_iter = 50
drm_history = douglas_rachford(x0, setA, setB, n_iter)
crm_history = tangent_space_crm(x0, setA, setB, n_iter)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.plot(drm_history[:, 0], drm_history[:, 1], 'o-', label='DRM', alpha=0.7)
plt.plot(crm_history[:, 0], crm_history[:, 1], 'x-', label='CRM (Tangent-Space)', alpha=0.7)
plt.plot(setA['center'][0], setA['center'][1], 'ro', label='Set A Center')
plt.plot(setB['center'][0], setB['center'][1], 'bo', label='Set B Center')
plt.legend()
plt.title('DRM vs CRM on Poincaré Ball')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
