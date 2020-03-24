Source files:
- src/map.py: Classes for occupancy grid map.
    - Map2D: Class for 2D occupancy grid map. By setting the range of x and y axis and resolution, this class automatically
    builds the grid map and provides methods to convert world coordinates to pixel indicies
- src/robot.py: Classes and functions for THOR robot.
    - Transform: Transform matrix generator. By creating an instance with the current THOR robot state, this class automatically build transform matrices between different frames.
    - undistort_(ir/rgb): undistort color and depth image with predefined calibration parameters
    - align_ir_rgb: align color and depth image with predefined extrinisic and parameters.
- src/slam.py: ParticleFilterSLAM class for experiment
    - **please refer to the notebook for usage**
    - update_particles
    - update_map_logodds
    - update_map_texture
- src/util.py: utilities for robot trajectory, occupancy map, and texture map visualization
    - plot_map: plot the map and trajectory
    - save_map_fig: save the map and trajectory to image file

Scripts:
- script/make_traindata.py: making training data pickle files, this script load robot data from different files with project provided function, align timestamps, transform lidar scan to [x, y, z] lidar frame, and save the data to a pickle file. The experiment notebooks all use the processed pickle file instead of raw robot data.
- script/make_video.py: make video from experiment output images

Experiment Notebooks:
- SLAM_<id>: The particle filter SLAM experiment on data set <id>.

