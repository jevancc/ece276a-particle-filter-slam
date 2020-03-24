import cv2
import numpy as np
import itertools


def yaw(rad):
    return np.array([[np.cos(rad), -np.sin(rad), 0],
                     [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


def pitch(rad):
    return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0],
                     [-np.sin(rad), 0, np.cos(rad)]])


def roll(rad):
    return np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)],
                     [0, np.sin(rad), np.cos(rad)]])


def lidar_scan_to_points(lidar_scan, lidar_scan_rlim=(0.93, 30)):
    lidar_scan_points = []
    for i in range(1081):
        r = lidar_scan[0, i]
        rmin, rmax = lidar_scan_rlim
        if rmin < r < rmax:
            rad = np.deg2rad(-135 + (i / 1081) * 270)
            lidar_scan_points.append([r * np.cos(rad), r * np.sin(rad), 0])
    return np.array(lidar_scan_points)


class Transform:

    def __init__(self,
                 x=None,
                 y=None,
                 theta=None,
                 neck_angle=None,
                 head_angle=None,
                 *args,
                 **kwargs):
        self._x = x
        self._y = y
        self._theta = theta
        self._neck_angle = neck_angle
        self._head_angle = head_angle

    @property
    def headTkinect(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.07],
                         [0, 0, 0, 1]])

    @property
    def headTlidar(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.15],
                         [0, 0, 0, 1]])

    @property
    def bodyThead(self):
        R = yaw(self._neck_angle) @ pitch(self._head_angle)
        P = np.array([[0, 0, 0.33]]).T
        return np.vstack([np.hstack([R, P]), np.array([0, 0, 0, 1])])

    @property
    def worldTbody(self):
        R = yaw(self._theta)
        P = np.array([[self._x, self._y, 0.93]]).T
        return np.vstack([np.hstack([R, P]), np.array([0, 0, 0, 1])])

    @property
    def opticalTregular(self):
        return np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64)

    @property
    def regularToptical(self):
        return np.linalg.inv(self.opticalTregular)

    @property
    def oTr(self):
        return self.opticalTregular

    @property
    def rTo(self):
        return self.regularToptical

    @property
    def hTk(self):
        return self.headTkinect

    @property
    def hTl(self):
        return self.headTlidar

    @property
    def bTh(self):
        return self.bodyThead

    @property
    def wTb(self):
        return self.worldTbody

    def chain(self, *transforms):
        ret = np.eye(4)
        for t in transforms:
            ret = ret @ getattr(self, t)
        return ret


def _extrinsics_ir_rgb():
    # The following define a transformation from the IR to the RGB frame:
    rgb_R_ir = np.array(
        [[0.99996855100876, 0.00589981445095168, 0.00529992291318184],
         [-0.00589406393353581, 0.999982024861347, -0.00109998388535087],
         [-0.00530631734715523, 0.00106871120747419, 0.999985350318977]])
    rgb_T_ir = np.array([[0.0522682, 0.0015192, -0.0006059]]).T  # meters

    return np.vstack(
        [np.hstack([rgb_R_ir, rgb_T_ir]),
         np.array([0, 0, 0, 1.0])])


EXTRINSICS_IR_RGB = _extrinsics_ir_rgb()


def _ir_calib():
    '''For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/'''
    #-- Focal length:
    fc = np.array([364.457362485643273, 364.542810626989194])
    #-- Principal point:
    cc = np.array([258.422487561914693, 202.487139940005989])
    #-- Skew coefficient:
    alpha_c = 0.000000000000000
    #-- Distortion coefficients:
    kc = np.array([
        0.098069182739161, -0.249308515140031, 0.000500420465085,
        0.000529487524259, 0.000000000000000
    ])
    #-- Focal length uncertainty:
    fc_error = np.array([1.569282671152671, 1.461154863082004])
    #-- Principal point uncertainty:
    cc_error = np.array([2.286222691982841, 1.902443125481905])
    #-- Skew coefficient uncertainty:
    alpha_c_error = 0.000000000000000
    #-- Distortion coefficients uncertainty:
    kc_error = np.array([
        0.012730833002324, 0.038827084194026, 0.001933599829770,
        0.002380503971426, 0.000000000000000
    ])
    #-- Image size: nx x ny
    nxy = np.array([512, 424])
    return {
        'fc': fc,
        'cc': cc,
        'ac': alpha_c,
        'kc': kc,
        'nxy': nxy,
        'fce': fc_error,
        'cce': cc_error,
        'ace': alpha_c_error,
        'kce': kc_error
    }


IR_CALIB = _ir_calib()


def _rgb_calib():
    '''For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/'''
    #-- Focal length:
    fc = np.array([525, 525.5])
    #-- Principal point:
    cc = np.array([524.5, 267])
    #-- Skew coefficient:
    alpha_c = 0.000000000000000
    #-- Distortion coefficients:
    kc = np.array([
        0.026147836868708, -0.008281285819487, -0.000157005204226,
        0.000147699131841, 0.000000000000000
    ])
    #-- Focal length uncertainty:
    fc_error = np.array([2.164397369394806, 2.020071561303139])
    #-- Principal point uncertainty:
    cc_error = np.array([3.314956924207777, 2.697606587350414])
    #-- Skew coefficient uncertainty:
    alpha_c_error = 0.000000000000000
    #-- Distortion coefficients uncertainty:
    kc_error = np.array([
        0.005403085916854, 0.015403918092499, 0.000950699224682,
        0.001181943171574, 0.000000000000000
    ])
    #-- Image size: nx x ny
    nxy = np.array([960, 540])
    return {
        'fc': fc,
        'cc': cc,
        'ac': alpha_c,
        'kc': kc,
        'nxy': nxy,
        'fce': fc_error,
        'cce': cc_error,
        'ace': alpha_c_error,
        'kce': kc_error
    }


RGB_CALIB = _rgb_calib()


def _undistort(img, calib):
    h, w = img.shape[:2]
    mtx = np.array([[calib['fc'][0], 0.0, calib['cc'][0]],
                    [0, calib['fc'][1], calib['cc'][1]], [0, 0, 1]])

    dst = cv2.undistort(img, mtx, calib['kc'], None)
    return dst


def undistort_rgb(img):
    return _undistort(img, RGB_CALIB)


def undistort_ir(img):
    return _undistort(img, IR_CALIB)


def align_ir_rgb(ir, rgb):
    h, w = ir.shape[:2]
    aligned = np.zeros((h, w, 6))
    for v, u in itertools.product(range(h), range(w)):
        z = ir[v, u] / 1000
        x = (u - IR_CALIB['cc'][0]) * z / IR_CALIB['fc'][0]
        y = (v - IR_CALIB['cc'][1]) * z / IR_CALIB['fc'][1]

        transformed = EXTRINSICS_IR_RGB @ np.array([[x, y, z, 1]]).T
        aligned[v, u, :3] = transformed[:3, 0]

    hrgb, wrgb = rgb.shape[:2]
    for v, u in itertools.product(range(h), range(w)):
        x = (aligned[v, u, 0] * RGB_CALIB['fc'][0] /
             aligned[v, u, 2]) + RGB_CALIB['cc'][0]
        y = (aligned[v, u, 1] * RGB_CALIB['fc'][1] /
             aligned[v, u, 2]) + RGB_CALIB['cc'][1]

        x, y = np.array([x, y]).astype(np.int32)

        if x < wrgb and y < hrgb and x >= 0 and y >= 0:
            aligned[v, u, 3:] = rgb[y, x, :]

    return aligned
