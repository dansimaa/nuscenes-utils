import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes as NuScenesDatabase


def get_lidar_timestamp(nusc, first_lidar_token):
    """
    Build a dictionary mapping lidar timestamps to they
    corresponding tokens.
    """
    timestamp_to_token = {}
    lidar_token = first_lidar_token    
    while lidar_token:
        lidar = nusc.get('sample_data', lidar_token)
        timestamp_to_token[lidar["timestamp"]] = lidar_token
        lidar_token = nusc.get('sample_data', lidar_token)["next"]        
    return timestamp_to_token


def find_closest_timestamp(lidar_timestamps, cam_timestamp):
    closest_timestamp = None
    min_difference = float('inf')  
    for timestamp in lidar_timestamps:
        difference = abs(timestamp - cam_timestamp)
        if difference < min_difference:
            closest_timestamp = timestamp
            min_difference = difference
    return closest_timestamp


def create_depth_map(pixel_data, depths, im):
    """
    Create the projected depth map image.
    """
    H, W, _  = np.array(im).shape
    depth_map = np.zeros((H, W))  
    for pixel_info, depth in zip(pixel_data.T, depths):
        x = int(pixel_info[0])
        y = int(pixel_info[1])
        if 0 <= x < W and 0 <= y < H:  
            depth_map[y, x] = depth          
    return depth_map


def main(args):
    """
    Project the LiDAR pointclouds for samples and sweeps frames.
    For sweeps the closest time stamp between camera frame and lidar is chosen
    to project the lidar pointcloud.
    """
    # Nuscenes    
    nusc = NuScenesDatabase(
        version=args.version,
        dataroot=args.input_dir,
        verbose=False,
    )
    with open(os.path.join(nusc.dataroot, f"{args.version}/scene.json")) as f:
        scene_json = json.load(f)
    scene_names = [elem["name"] for elem in scene_json]
    cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
    cameras = ["CAM_" + camera for camera in cameras]

    # Create the output folders
    for camera in cameras: 
        for frametype in ["samples", "sweeps"]:        
            output_path = os.path.join(nusc.dataroot, f"{frametype}/projected_lidar", camera)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

    for scene_name in scene_names:        
        # Current scene
        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
        scene = nusc.get('scene', scene_token)
        first_sample = nusc.get('sample', scene['first_sample_token'])
        
        # LiDAR timestamps
        first_lidar_token = first_sample['data']["LIDAR_TOP"]
        lidar_timestamp_to_token = get_lidar_timestamp(nusc, first_lidar_token)
        lidar_timestamps = list(lidar_timestamp_to_token.keys())

        for camera in tqdm(cameras, desc=f"Processing {scene_name}"):
            # Initialize the camera tokens
            first_camera_token = first_sample['data'][camera]
            camera_token = first_camera_token

            while camera_token:                
                # Get the closest lidar pointcloud
                cam = nusc.get('sample_data', camera_token)
                cam_timestamp = cam["timestamp"]
                closest_lidar_timestamp = find_closest_timestamp(lidar_timestamps, cam_timestamp)
                lidar_token = lidar_timestamp_to_token[closest_lidar_timestamp]

                points, coloring, im = nusc.explorer.map_pointcloud_to_image(
                    lidar_token,
                    camera_token,
                    render_intensity=False,
                    show_lidarseg=False,
                    filter_lidarseg_labels=[22, 23, 24],
                    lidarseg_preds_bin_path=None,
                    show_panoptic=False
                )
                depth_map = create_depth_map(points, coloring, im)

                # Save depth map as npz
                cam_path = cam['filename']
                save_path = os.path.join(nusc.dataroot, cam_path.split("/")[0],
                                         "projected_lidar", camera)
                np.savez(os.path.join(save_path, cam_path.split("/")[2].replace(".jpg", ".npz")),
                         depth_map)
                
                # Save projected lidar as an rgb image
                if args.save_image:                                 
                    save_path_img = os.path.join(save_path, cam_path.split("/")[2])
                    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
                    fig.canvas.set_window_title("")
                    ax.imshow(im)
                    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
                    ax.axis('off')
                    plt.savefig(save_path_img, bbox_inches='tight', pad_inches=0, dpi=200)

                # Next camera token
                camera_token = nusc.get('sample_data', camera_token)["next"]


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Project the lidar pointcloud on the images")
    parser.add_argument("--input_dir", 
                        type=str, 
                        required=True,
                        help="Nuscenes root directory")
    parser.add_argument("--version", 
                        type=str, 
                        default="v1.0-mini",
                        help="Nuscenes version")
    parser.add_argument("--save_image", 
                        action="store_true", 
                        default=False,
                        help="Save the rgb image")
    args = parser.parse_args()

    main(args)