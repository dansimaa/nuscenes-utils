import argparse
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn import linear_model
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def plot_pc(xyz):
    """
    Given an input point cloud as a np.array (n_points, 3), 
    plot it using open3d, adding the coordinate frame.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    # o3d.visualization.draw_geometries([point_cloud])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    vis.run()
    vis.destroy_window()


def generate_grid(x_min, x_max, y_min, y_max, n_samples):
    x_values = np.linspace(x_min, x_max, num=n_samples)
    y_values = np.linspace(y_min, y_max, num=n_samples)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    return grid
   

def from_lidar_to_cam(pc, nusc, pointsensor, cam):
    """
    Transform a point cloud from the LiDAR coordinate system to the camera coordinate system,
    and subsequently to the pixel coordinate system for a given NuScenes sample.
    """
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)      

    depths = pc.points[2, :]
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    
    # Remove points behind the camera
    min_dist = 1.0
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    points = points[:, mask]
    depths = depths[mask]
    return points, depths


def main(args):
    """
    Utilizing LiDAR segmentation labels, this function generates depth maps 
    for the road plane in Nuscenes samples. It accomplishes this by fitting 
    a plane to the road point cloud and subsequently performing perspective 
    projection from the LiDAR frame to the image plane.
    """
    # Nuscenes    
    nusc = NuScenesDatabase(
        version=args.version,
        dataroot=args.input_dir,
        verbose=False,
    )
    cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
    cameras = ["CAM_" + camera for camera in cameras]

    # Create the output folders
    for camera in cameras:         
        output_path = os.path.join(nusc.dataroot, "samples/road_depth_plane", camera)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        vis_path = os.path.join(nusc.dataroot, "samples/road_depth_plane_vis", camera)
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    
    # Iterate over samples
    for sample in tqdm(nusc.sample):
        sample_record = nusc.get('sample', sample["token"])
        lidar_token = sample_record['data']["LIDAR_TOP"]

        for camera in cameras:
            camera_token = sample_record['data'][camera]
            cam = nusc.get('sample_data', camera_token)
            cam_path = cam['filename']
            im = Image.open(os.path.join(nusc.dataroot, cam_path))
            H, W, _  = np.array(im).shape
    
            # LiDAR
            pointsensor = nusc.get('sample_data', lidar_token)
            lidar_path = os.path.join(nusc.dataroot, pointsensor['filename'])
            pc = LidarPointCloud.from_file(lidar_path)

            # LiDAR Segmentation
            lidarseg_labels_path = os.path.join(nusc.dataroot, 
                                                 nusc.get('lidarseg', lidar_token)['filename'])
            points_label = np.fromfile(lidarseg_labels_path, dtype=np.uint8)            

            # Road mask from the road labels 
            road_labels_name = [category for category in nusc.lidarseg_name2idx_mapping if category.startswith("flat")]
            road_labels_idx = [nusc.lidarseg_name2idx_mapping.get(name) for name in road_labels_name]
            road_mask = np.isin(points_label, road_labels_idx)

            # Road point cloud
            lidar_pc = pc.points[:3]
            road_pointcloud = lidar_pc[:, road_mask].T

            # Fit 3D plane using the road points 
            xy_train = road_pointcloud[:, :2]
            z_train = road_pointcloud[:, 2]
            lin_model = linear_model.LinearRegression()
            lin_model.fit(xy_train, z_train)

            # Generate the road plane
            x_min, x_max = min(road_pointcloud[:, 0]), max(road_pointcloud[:, 0])
            y_min, y_max = min(road_pointcloud[:, 1]), max(road_pointcloud[:, 1])
            xy_points = generate_grid(x_min, x_max, y_min, y_max, args.n_samples)
            z_points = lin_model.predict(xy_points)
            plane_pc = np.column_stack((xy_points, z_points))
            # plot_pc(plane_pc)

            # Update LidarPointCloud to perform transformation
            pc.points = plane_pc.T

            # Get the pixel coordinate system points and depth values
            points, depths = from_lidar_to_cam(pc, nusc, pointsensor, cam)

            # Build the road depth map
            depth_map = np.zeros((H, W))  
            for pixel_info, depth in zip(points.T, depths):
                x = int(pixel_info[0])
                y = int(pixel_info[1])
                if 0 <= x < W and 0 <= y < H:  
                    depth_map[y, x] = depth        
                        
            # Save the road depth plane as npz and jpg
            file_name = cam_path.split("/")[2]
            plane_path = os.path.join(nusc.dataroot, "samples/road_depth_plane", 
                                      camera, file_name.replace(".jpg", ".npz"))                
            plane_path_vis = os.path.join(nusc.dataroot, "samples/road_depth_plane_vis",
                                          camera, file_name)
            
            np.savez(plane_path, depth_map)
            depth_map_color = plt.get_cmap('magma')(depth_map, bytes=True)[..., :3]
            depth_img = Image.fromarray(depth_map_color)
            depth_img.save(plane_path_vis)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the depth road plane for Nuscenes")
    parser.add_argument("--input_dir", 
                        type=str, 
                        required=True,
                        help="Nuscenes root directory")
    parser.add_argument("--version", 
                        type=str, 
                        default="v1.0-mini",
                        help="Nuscenes version")
    parser.add_argument("--n_samples", 
                        type=int, 
                        default=5000,
                        help="Number of points to build the plane grid")
    args = parser.parse_args()

    main(args)
