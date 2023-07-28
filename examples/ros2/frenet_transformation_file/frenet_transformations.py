import torch
import numpy as np
from scipy.interpolate import CubicSpline, BSpline, splrep, splprep, splev
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

def global_traj(global_path, dt=0.004):
    """
    This function converts the global path to frenet frames.
    Parameters:
    global_path: The global path which is to be transformed to frenet frame coordinate system
    dt: The resolution of the path in frenet frame. Higher the resolution, better will accuracy of transformation
    We first interpolate through the global path so that it has a resolution equal to dt. The we transform the interpolated path to frenet frame coordinate frame.
    """

    #   Stores array whos size is equal to the length of the global path
    time_array = np.arange(0, global_path.shape[0])
    g_path = np.zeros(global_path.shape)
    g_path[:,0] = global_path[:,0] 
    g_path[:,1] = global_path[:,1] 

    #   Interpolate through the global path to increase its resolution, defined by the parameter dt
    # cs = CubicSpline(time_array, g_path)
    sp_x = splrep(time_array, g_path[:,0], k=3, s=0.0)
    sp_y = splrep(time_array, g_path[:,1], k=3, s=0.0)
    xs = np.arange(0, g_path.shape[0], dt)
    # g_path = cs(xs)
    g_path = np.vstack((splev(xs, sp_x, ext=3), splev(xs, sp_y, ext=3))).T

    #   Calculate the path orientation for the global path before the transformation
    x_diff = np.diff(g_path[:,0])
    y_diff = np.diff(g_path[:,1])
    theta = np.arctan2(y_diff, x_diff)

    #    Transform the path to the frenet frame coordinate frame, by taking the euclidean distance between the interpolated higher resolution global path and taking the cumulation sum of the eucliden distances. 
    new_g_path = np.sqrt(x_diff**2 + y_diff**2)
    new_g_path = np.cumsum(new_g_path)
    #   Here, we are assuming the vehile has an orientation of 90 degree, hence new_g_path is in column 2, which is the y axis in frenet frame. To change the axis in frenet frame so that vehicle starts at an orientation of 0 degree, swap the variables below.
    new_g_path = np.vstack(((g_path[0][0])*np.ones((new_g_path.shape[0])), new_g_path)).T
    

    return new_g_path, g_path, theta

def global_to_frenet(obstacle_array, new_global_path, g_path):
    """
    This function transforms the obstacles(or any object of interest in the environment) in the global frame to the frenet frame coordinate system.
    Parameters:
    obstacle_array: An array of [x,y] coordinates which need to be transformed from global frame to the frenet frame coordinate system.
    new_global_path: Global path in frenet frame
    g_path: Global path in global frame
    """
    path_arc_lengths = new_global_path[:, 1]
    frenet_obs = []
    for i in range(obstacle_array.shape[0]):
        # We calculate the distance of the point from each point in the global path
        dists_from_path = np.linalg.norm(obstacle_array[i] - g_path, axis=1)
        # We get tge index of the minimum distance
        nearest_point_idx = np.argmin(dists_from_path)
        nearest_point_to_obs = path_arc_lengths[nearest_point_idx]
        min_dist_from_path = np.sign(obstacle_array[i][0])*dists_from_path[nearest_point_idx] + new_global_path[i][0]
        frenet_obs.append([min_dist_from_path, nearest_point_to_obs])
        
    return np.array(frenet_obs)

def frenet_to_global_with_traj(trajectory, new_global_path, g_path, dt=0.1):
    """
    This function converts the trajectry given by the optimizer from frenet frame coordinate system to gobal frame.
    Parameters:
    trajectory: The trajectory given bythe optimizer which needs to be transformed from frenet frame coordinate system to gobal frame.
    new_global_path: Global path in frenet frame
    g_path: Global path in global frame
    dt: This is the time resolution for the optimizer trajectory, which is 0.1s.
    """
    #   We only consider the x and y coordinates of the global path
    new_global_path = new_global_path[:-2]
    #   We calculate the the distance between each point in the trajectory to each point in the frenet global path using scipy cdist, which takes < 5 ms for an array of size 10,000x10,000
    traj_dist_from_g_path = cdist(trajectory[:, :2], new_global_path)
    #   We then get the nearest point in the frenet global path corresponding to the position in the trajectory, for transformation to global frame
    nearest_point_idxs = np.argmin(traj_dist_from_g_path, axis=1)
    #   We add 1 to the indices for calculation of theta
    next_nearest_points_idxs = nearest_point_idxs + 1

    #   We calculate the theta in the global path for the indices obtained above
    g_path_theta = np.arctan2(g_path[next_nearest_points_idxs][:,1] - g_path[nearest_point_idxs][:,1], g_path[next_nearest_points_idxs][:,0] - g_path[nearest_point_idxs][:,0])

    #   We get the minimum distance of the trajectory from the frenet global path usinig the shortest distance indices obtained above
    dists = []
    for i in range(traj_dist_from_g_path.shape[0]):
        dists.append(traj_dist_from_g_path[i, nearest_point_idxs[i]])
    dists = np.array(dists)
    coord_x = []
    coord_y = []
    #   We then using the shortes distance obtained above to get the corresponding point in the global frame for a position in the trajectory. We use polar coordinates for that, where radius is the shortes distance obtained above and angle is the difference between global path theta and trajectory theta.
    for i in range(dists.shape[0]):
        #   Since we are assuming the orientation of the vehicle as 90 and the vehicle moving in y axis, value of the variable val is 90 degree. Please change the values for operation along x axis
        val = np.pi/2
        if new_global_path[nearest_point_idxs[i],0]<trajectory[i,0]:
            val = -np.pi/2 
        coord_x.append(dists[i]*np.cos(g_path_theta[i]+val) + g_path[nearest_point_idxs[i], 0])
        coord_y.append(dists[i]*np.sin(g_path_theta[i]+val) + g_path[nearest_point_idxs[i], 1])
    coord_theta =  g_path_theta - (np.pi/2 - trajectory[:, 2])
    coord_x = np.array(coord_x)
    coord_y = np.array(coord_y)
    omega = np.diff(coord_theta)/dt
    velocity = (np.diff(coord_x)**2 + np.diff(coord_y)**2)**0.5/dt

    return torch.as_tensor(np.vstack((coord_x, coord_y))).T, torch.as_tensor(np.vstack((velocity, omega))).T

    # plt.scatter(g_path[nearest_point_idxs][:, 0], g_path[nearest_point_idxs][:, 1], c='b')
    # plt.scatter(new_global_path[nearest_point_idxs][:, 0], new_global_path[nearest_point_idxs][:, 1], c='g')
    # plt.plot(coord_x, coord_y, 'y')
    # plt.plot(trajectory[:,0], trajectory[:,1], 'r')
    # plt.show()
    # quit()
    
    # # mean_controls[:, 1] = controls[:,1] + theta_dot[nearest_point_idxs]
    # # return mean_controls