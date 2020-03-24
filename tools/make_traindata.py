import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import click

sys.path.append('..')
from data.load_data import *


@click.command()
@click.option('-i', '--id', required=True, help='Training data ID')
@click.option('--datadir', default='../data', help='Training data directory')
@click.option('--camera',
              is_flag=True,
              default=False,
              help='Bundle camera data')
def main(id, datadir, camera):
    JOINT_DATA_PATH = os.path.join(datadir, f'joint/train_joint{id}')
    LIDAR_DATA_PATH = os.path.join(datadir, f'lidar/train_lidar{id}')

    print('JOINT_DATA_PATH::', JOINT_DATA_PATH)
    print('LIDAR_DATA_PATH::', LIDAR_DATA_PATH)

    jdata = get_joint(JOINT_DATA_PATH)
    jdata['ts'] = (jdata['ts'] * 10000).astype(np.int64)

    ldata = get_lidar(LIDAR_DATA_PATH)
    for lidar_data in ldata:
        lidar_data['t'] = (lidar_data['t'] * 10000).astype(np.int64)

    if camera:
        CAMERA_DEPTH_DATA_PATH = os.path.join(datadir, f'cam/DEPTH_{id}')
        CAMERA_RGB_DATA_PATH = os.path.join(datadir, f'cam/RGB_{id}')

        print('CAMERA_DEPTH_DATA_PATH::', CAMERA_DEPTH_DATA_PATH)
        print('CAMERA_RGB_DATA_PATH::', CAMERA_RGB_DATA_PATH)

        ddata = get_depth(CAMERA_DEPTH_DATA_PATH)
        for d in ddata:
            d['t'] = np.int64(d['t'] * 10000)

        rgbdata = get_rgb(CAMERA_RGB_DATA_PATH)
        for d in rgbdata:
            d['t'] = np.int64(d['t'] * 10000)
    else:
        ddata = []
        rgbdata = []

    joint_i, depth_i, rgb_i = 0, 0, 0

    data = []
    for i, lidar_data in enumerate(ldata):
        while joint_i < jdata['ts'].shape[-1] and jdata['ts'][
                0, joint_i] <= lidar_data['t'][0, 0]:
            joint_i += 1
        joint_i = max(joint_i - 1, 0)

        while depth_i < len(ddata) and ddata[depth_i]['t'] <= lidar_data['t'][
                0, 0]:
            depth_i += 1
        depth_i = max(depth_i - 1, 0)

        while rgb_i < len(rgbdata) and rgbdata[rgb_i]['t'] <= lidar_data['t'][
                0, 0]:
            rgb_i += 1
        rgb_i = max(rgb_i - 1, 0)

        lidar_scan_points = []
        for i in range(1081):
            distance = lidar_data['scan'][0, i]
            if distance > 0.93 and distance < 30:
                angle = np.deg2rad(-135 + (i / 1081) * 270)
                lidar_scan_points.append(
                    [distance * np.cos(angle), distance * np.sin(angle), 0])

        lidar_scan_points = np.array(lidar_scan_points)

        data_t = {
            't': lidar_data['t'][0, 0],
            't_joint': jdata['ts'][0, joint_i],
            'lidar_delta_pose': lidar_data['delta_pose'],
            'lidar_scan': lidar_data['scan'],
            'lidar_scan_points': lidar_scan_points,
            'joint_head_angles': jdata['head_angles'][:, joint_i],
            'neck_angle': jdata['head_angles'][0, joint_i],
            'head_angle': jdata['head_angles'][1, joint_i]
        }
        if rgb_i < len(rgbdata) and rgbdata[rgb_i]['t'] <= lidar_data['t'][0,
                                                                           0]:
            data_t['image'] = rgbdata[rgb_i]['image']
            data_t['t_image'] = rgbdata[rgb_i]['t']
        else:
            data_t['image'] = None
            data_t['t_image'] = -1

        if depth_i < len(ddata) and ddata[depth_i]['t'] <= lidar_data['t'][0,
                                                                           0]:
            data_t['depth'] = ddata[depth_i]['depth']
            data_t['t_depth'] = ddata[depth_i]['t']
        else:
            data_t['depth'] = None
            data_t['t_depth'] = -1

        data.append(data_t)

    assert len(data) == len(ldata)

    OUTPUT_PATH = os.path.join(datadir, f'train{id}.pkl')
    print('OUTPUT::', OUTPUT_PATH)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
