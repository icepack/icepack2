r"""Physical constants. The factors of seconds -> year conversion and 1e-6 are
because we use a unit system of MPa, meters, and years."""

__all__ = [
    "ice_density",
    "water_density",
    "gravity",
    "glen_flow_law",
    "weertman_sliding_law",
]

year = 365.25 * 24 * 60 * 60
ice_density = 917 / year**2 * 1e-6
water_density = 1024 / year**2 * 1e-6
gravity = 9.81 * year**2
glen_flow_law = 3.0
weertman_sliding_law = 3.0
