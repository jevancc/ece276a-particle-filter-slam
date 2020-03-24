import numpy as np


class Map2D:

    def __init__(self,
                 xlim=(-20, 20),
                 ylim=(-20, 20),
                 resolution=0.05,
                 dtype=np.float64):
        self.resolution = resolution
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.xsize = int(np.ceil((self.xmax - self.xmin) / self.resolution + 1))
        self.ysize = int(np.ceil((self.ymax - self.ymin) / self.resolution + 1))
        self._map = np.zeros((self.xsize, self.ysize), dtype=dtype)

    @property
    def data(self):
        return self._map

    @data.setter
    def data(self, d):
        assert d.shape == self._map.shape
        self._map = d

    def in_map(self, coordinates):
        return np.logical_and(
            np.logical_and(self.xmin <= coordinates[:, 0],
                           coordinates[:, 0] <= self.xmax),
            np.logical_and(self.ymin <= coordinates[:, 1],
                           coordinates[:, 1] <= self.ymax))

    def coordinate_to_index(self, coordinates):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)

        return np.hstack([
            np.ceil((coordinates[:, 0] - self.xmin) / self.resolution).reshape(
                -1, 1),
            np.ceil((coordinates[:, 1] - self.ymin) / self.resolution).reshape(
                -1, 1),
        ]).astype(np.int32)
