import numpy as np


def to_one_hot(x, num_classes=None, dtype=np.float32):

    if num_classes is None:
        num_classes = np.unique(x).size
        print("found classes", np.unique(x).astype(int))

    if num_classes == 1:
        num_classes = 2

    one_hot = np.eye(num_classes)[x.squeeze().astype(int)]

    return one_hot


def vec2ang(v):
    x, y, z = np.asarray(v).T
    phi = np.arctan2(y, x)
    theta = np.pi / 2 - np.arctan2(z, (x**2 + y**2)**.5)
    return np.rad2deg(phi), np.rad2deg(theta)


def ang2vec(Azimuth, Zenith, deg=True):
    """ Get 3-vector from spherical angles.
    Args:
        phi (Azimuth): azimuth (pi, -pi), 0 points in x-direction, pi/2 in y-direction
        zenith (Zenith): zenith (0, pi), 0 points in z-direction
    Returns:
        array of 3-vectors
    """
    if deg is True:
        Zenith = np.deg2rad(Zenith)
        Azimuth = np.deg2rad(Azimuth)

    x = np.sin(Zenith) * np.cos(Azimuth)
    y = np.sin(Zenith) * np.sin(Azimuth)
    z = np.cos(Zenith)
    return np.array([x, y, z]).T
