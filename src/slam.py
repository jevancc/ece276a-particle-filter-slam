import cv2
import itertools
import contextlib
import numpy as np
from .robot import *
from .map import *


class ParticleFilterSLAM:

    def __init__(self,
                 n_particles,
                 n_efficient_threshold,
                 predict_noise_sigma,
                 map_xlim,
                 map_ylim,
                 map_resolution,
                 map_logodds_lim,
                 map_logodds_occupied_diff,
                 map_logodds_free_diff,
                 map_logodds_occupied_threshold,
                 map_texture_undistort_image=True,
                 map_texture_update_alpha=0.2,
                 *args,
                 **kwargs):

        self.n_particles = n_particles
        self.n_efficient_threshold = n_efficient_threshold

        self.map_logodds_min = map_logodds_lim[0]
        self.map_logodds_max = map_logodds_lim[1]
        self.map_logodds_occupied_diff = map_logodds_occupied_diff
        self.map_logodds_free_diff = map_logodds_free_diff
        self.map_logodds_occupied_threshold = map_logodds_occupied_threshold

        self.predict_noise_sigma = predict_noise_sigma

        self.particles = np.zeros((n_particles, 3))
        self.weights = np.ones((n_particles)) / n_particles
        self.map_logodds = Map2D(xlim=map_xlim,
                                 ylim=map_ylim,
                                 resolution=map_resolution)

        self.map_texture = np.zeros(
            (self.map_logodds.xsize, self.map_logodds.ysize, 3),
            dtype=np.float64)
        self.map_texture_update_alpha = map_texture_update_alpha
        self.map_texture_undistort_image = map_texture_undistort_image

        self.current_robot_data = {}

    @property
    def robot_state(self):
        return np.sum(self.weights.reshape(-1, 1) * self.particles, axis=0)

    @contextlib.contextmanager
    def set_robot_data(self, **data):
        self.current_robot_data = data
        yield self
        self.current_robot_data = {}

    @property
    def map_binary(self, occupied_threshold=None):
        if occupied_threshold is None:
            occupied_threshold = self.map_logodds_occupied_threshold
        return (self.map_logodds.data > occupied_threshold).astype(np.int32)

    @property
    def map_prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.map_logodds.data))

    @property
    def map_texture_trimmed(self):
        gamma = self.map_prob
        not_ground_mask = (gamma > 0.0).reshape(-1)
        wall_mask = (gamma > 0.8).reshape(-1)

        mp = self.map_texture_raw.reshape(-1, 3)
        mp[not_ground_mask, :] = 0
        mp[wall_mask, :] = np.array([0, 255, 0])
        mp = mp.reshape(*gamma.shape, 3)
        return mp

    @property
    def map_texture_raw(self):
        return self.map_texture.astype(np.int32)

    def in_map(self, coordinates):
        return self.map_logodds.in_map(coordinates)

    def coordinate_to_map_index(self, coordinates):
        return self.map_logodds.coordinate_to_index(coordinates)

    def resample(self):
        weights = self.weights
        n = self.n_particles
        indices = []
        C = [0.0] + [np.sum(weights[:i + 1]) for i in range(n)]
        u0, j = np.random.random(), 0
        for u in [(u0 + i) / n for i in range(n)]:
            while u > C[j]:
                j += 1
            indices.append(j - 1)
        return indices

    def predict_particles(self, u, noise_sigma=None):
        if noise_sigma is None:
            noise_sigma = self.predict_noise_sigma

        u = np.random.multivariate_normal(np.array(u), noise_sigma,
                                          self.n_particles)
        self.particles += u
        return u

    def _prepare_lidar_scan_points(self, lidar_scan_rlim):
        if 'lidar_scan_points' not in self.current_robot_data:
            self.current_robot_data['lidar_scan_points'] = lidar_scan_to_points(
                self.current_robot_data['lidar_scan'], lidar_scan_rlim)
        return self.current_robot_data['lidar_scan_points']

    def update_particles(self, lidar_scan_rlim=(0.93, 30)):
        lidar_scan_points = self._prepare_lidar_scan_points(lidar_scan_rlim)

        lidar_scan_points = np.hstack(
            [lidar_scan_points,
             np.ones((lidar_scan_points.shape[0], 1))])

        t = Transform(neck_angle=self.current_robot_data.get('neck_angle'),
                      head_angle=self.current_robot_data.get('head_angle'))
        lidar_scan_points_b = t.chain('bTh', 'hTl') @ lidar_scan_points.T

        corr = np.zeros_like(self.weights)
        map_binary = self.map_binary

        for i, p in enumerate(self.particles):
            t = Transform(x=p[0], y=p[1], theta=p[2])
            lidar_scan_points_w = t.wTb @ lidar_scan_points_b
            lidar_scan_points_w = lidar_scan_points_w.T[:, :3]

            # Filter lidar scan points above ground
            lidar_scan_points_w = lidar_scan_points_w[
                lidar_scan_points_w[:, 2] > 0.1, :]

            # Filter lidar scan points in the map for updating particles
            lidar_scan_points_w = lidar_scan_points_w[
                self.map_logodds.in_map(lidar_scan_points_w), :]
            lidar_scan_points_indices = self.map_logodds.coordinate_to_index(
                lidar_scan_points_w)

            # 5x5 Map Correlation
            bias = {}
            c = np.zeros(25)
            for j, (bx, by) in enumerate(
                    itertools.product(range(-2, 3), range(-2, 3))):
                bias[j] = (bx, by)
                try:
                    c[j] = np.sum(
                        map_binary[lidar_scan_points_indices[:, 1] + by,
                                   lidar_scan_points_indices[:, 0] + bx])
                except Exception:
                    c[j] = -1

            # Update particle to local state with max correlation
            idx = np.argmax(c)
            self.particles[i, 0] += bias[idx][0] * self.map_logodds.resolution
            self.particles[i, 1] += bias[idx][1] * self.map_logodds.resolution
            corr[i] = c.max()

        log_weights = np.log(self.weights) + corr
        log_weights -= log_weights.max() + np.log(
            np.exp(log_weights - log_weights.max()).sum())
        self.weights = np.exp(log_weights)

        n_eff = 1 / np.sum(self.weights**2)

        if n_eff <= self.n_efficient_threshold:
            idx = self.resample()
            self.particles = self.particles[idx, :]
            self.weights = self.weights[idx]
            self.weights /= self.weights.sum()

        return corr, n_eff

    def update_map_logodds(self, robot_state=None, lidar_scan_rlim=(0.93, 30)):
        lidar_scan_points = self._prepare_lidar_scan_points(lidar_scan_rlim)

        if robot_state is None:
            robot_state = self.robot_state

        lidar_scan_points = np.hstack(
            [lidar_scan_points,
             np.ones((lidar_scan_points.shape[0], 1))])

        t = Transform(x=robot_state[0],
                      y=robot_state[1],
                      theta=robot_state[2],
                      neck_angle=self.current_robot_data.get('neck_angle'),
                      head_angle=self.current_robot_data.get('head_angle'))

        lidar_scan_points_w = t.chain('wTb', 'bTh', 'hTl') @ lidar_scan_points.T
        lidar_scan_points_w = lidar_scan_points_w.T[:, :3]

        # Filter lidar scan points above ground
        lidar_scan_points_w = lidar_scan_points_w[
            lidar_scan_points_w[:, 2] > 0.1, :]

        # Filter lidar scan points in the map for updating log-odds
        lidar_scan_points_w = lidar_scan_points_w[
            self.map_logodds.in_map(lidar_scan_points_w), :]

        lidar_scan_points_indices = self.map_logodds.coordinate_to_index(
            lidar_scan_points_w)

        self.map_logodds.data += cv2.drawContours(
            image=np.zeros_like(self.map_logodds.data),
            contours=[
                lidar_scan_points_indices.reshape((-1, 1, 2)).astype(np.int32)
            ],
            contourIdx=-1,
            color=self.map_logodds_free_diff,
            thickness=-1)
        self.map_logodds.data[
            lidar_scan_points_indices[:, 1],
            lidar_scan_points_indices[:,
                                      0]] += self.map_logodds_occupied_diff - self.map_logodds_free_diff

        self.map_logodds.data = np.clip(self.map_logodds.data,
                                        self.map_logodds_min,
                                        self.map_logodds_max)

    def update_map_texture(self, robot_state=None):
        if robot_state is None:
            robot_state = self.robot_state

        depth = self.current_robot_data.get('depth')
        image = self.current_robot_data.get('image')
        if depth is not None and image is not None:
            print('Updating map texture......')
            if self.map_texture_undistort_image:
                depth = undistort_ir(depth)
                image = undistort_rgb(image)

            t = Transform(
                x=robot_state[0],
                y=robot_state[1],
                theta=robot_state[2],
                neck_angle=self.current_robot_data.get('neck_angle'),
                head_angle=self.current_robot_data.get('head_angle'),
            )
            aligned = align_ir_rgb(depth, image).reshape(-1, 6)
            rgb_data = aligned[:, 3:]
            points = np.hstack([aligned[:, :3], np.ones((aligned.shape[0], 1))])
            points_w = t.chain('wTb', 'bTh', 'hTk', 'rTo') @ points.T
            points_w = points_w.T[:, :3]

            in_map_filter = self.in_map(points_w)
            points_w = points_w[in_map_filter, :]
            rgb_data = rgb_data[in_map_filter, :]

            ground_points_mask = np.logical_and(points_w[:, 2] < 0.1,
                                                points_w[:, 2] > -0.1)
            dark_color_mask = np.logical_and(
                np.logical_and(rgb_data[:, 0] > 20, rgb_data[:, 1] > 20),
                rgb_data[:, 2] > 20)

            mask = np.logical_and(dark_color_mask, ground_points_mask)
            points_w, rgb_data = points_w[mask, :], rgb_data[mask, :]

            points_indices = self.coordinate_to_map_index(points_w[:, [1, 0]])
            self.map_texture[points_indices[:, 0], points_indices[:, 1], :] *= (
                1.0 - self.map_texture_update_alpha)
            self.map_texture[
                points_indices[:, 0],
                points_indices[:,
                               1], :] += self.map_texture_update_alpha * rgb_data[:, :]
