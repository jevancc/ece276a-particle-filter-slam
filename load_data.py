"""""" """""" """""" """""" """""" """""" """""" """
DESCRIPTION
: In this file, you can load .mat file data in python dictionary format.
  The output of the "get_joint" function is a dictionary with eight different data (read data description for details). Each dictionary is an array with the same length.
  The "get_joint_index" function returns joint ID number.
  The output of the "get_lidar" function is an array with dictionary elements. The length of the array is the length of data.   
  The output of the "get_rgb" function is an array with dictionary elements. The length of the array is the length of data.
  The output of the "get_depth" function is an array with dictionary elements. The length of the array is the lenght of data.
	The replay_* functions help you to visualize and understand the lidar, depth, and rgb data. 
""" """""" """""" """""" """""" """""" """""" """"""

#import pickle
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def get_joint(file_name):
    key_names_joint = ['ts', 'head_angles']
    data = io.loadmat(file_name + ".mat")
    joint = {kn: data[kn] for kn in key_names_joint}
    return joint


def get_lidar(file_name):
    data = io.loadmat(file_name + ".mat")
    lidar = []
    for m in data['lidar'][0]:
        tmp = {}
        tmp['t'] = m[0][0][0]
        nn = len(m[0][0])
        if (nn != 3):
            raise ValueError("different length!")
        tmp['delta_pose'] = m[0][0][nn - 1]
        tmp['scan'] = m[0][0][nn - 2]
        lidar.append(tmp)
    return lidar


def replay_lidar(lidar_data):
    # lidar_data type: array where each array is a dictionary with a form of 't','pose','res','rpy','scan'
    theta = np.arange(0, 270.25, 0.25) * np.pi / float(180)

    for i in range(0, len(lidar_data), 1000):
        for (k, v) in enumerate(lidar_data[i]['scan'][0]):
            if v > 30:
                lidar_data[i]['scan'][0][k] = 0.0

    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, lidar_data[i]['scan'][0])
    ax.set_rmax(10)
    ax.set_rticks([2, 4])  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Lidar scan data", va='bottom')
    plt.show()


def get_rgb(folder_name):
    n_rgb = len(os.listdir(folder_name)) - 1
    rgb = []
    time_file = open(os.path.join(folder_name, "timestamp.txt"))
    for i in range(n_rgb):
        rgb_img = cv2.imread(os.path.join(folder_name, "%d.jpg" % (i + 1)))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        time = time_file.readline().split()
        rgb_dict = {'image': rgb_img, 't': float(time[1])}
        rgb.append(rgb_dict)
    return rgb


def replay_rgb(rgb_data):
    for k in range(len(rgb_data)):
        R = rgb_data[k]['image']
        R = np.flip(R, 1)
        plt.imshow(R)
        plt.draw()
        plt.pause(0.01)


def get_depth(folder_name):
    n_depth = len(os.listdir(folder_name)) - 1
    depth = []
    time_file = open(os.path.join(folder_name, "timestamp.txt"))
    for i in range(n_depth):
        depth_img = cv2.imread(os.path.join(folder_name, "%d.png" % (i + 1)),
                               -1)
        time = time_file.readline().split()
        depth_dict = {'depth': depth_img, 't': float(time[1])}
        depth.append(depth_dict)
    return depth


def replay_depth(depth_data):
    DEPTH_MAX = 4500
    DEPTH_MIN = 400
    for k in range(len(depth_data)):
        D = depth_data[k]['depth']
        D = np.flip(D, 1)
        for r in range(len(D)):
            for (c, v) in enumerate(D[r]):
                if (v <= DEPTH_MIN) or (v >= DEPTH_MAX):
                    D[r][c] = 0.0
        plt.imshow(D)
        plt.draw()
        plt.pause(0.01)


def getExtrinsics_IR_RGB():
    # The following define a transformation from the IR to the RGB frame:
    rgb_R_ir = np.array(
        [[0.99996855100876, 0.00589981445095168, 0.00529992291318184],
         [-0.00589406393353581, 0.999982024861347, -0.00109998388535087],
         [-0.00530631734715523, 0.00106871120747419, 0.999985350318977]])
    rgb_T_ir = np.array([0.0522682, 0.0015192, -0.0006059])  # meters
    return {'rgb_R_ir': rgb_R_ir, 'rgb_T_ir': rgb_T_ir}


def getIRCalib():
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


def getRGBCalib():
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


if __name__ == "__main__":
    j0 = get_joint("joint/train_joint0")
    l0 = get_lidar("lidar/train_lidar0")
    r0 = get_rgb("cam/RGB_0")
    d0 = get_depth("cam/DEPTH_0")
    exIR_RGB = getExtrinsics_IR_RGB()
    IRCalib = getIRCalib()
    RGBCalib = getRGBCalib()

    # visualize data
    replay_lidar(l0[:500])
    replay_rgb(r0[:20])
    replay_depth(d0[:5])
