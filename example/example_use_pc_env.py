import os
from time import time

import numpy as np
import open3d as o3d

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.real_world import task_setting

from simple_pc import SimplePointCloud

if __name__ == '__main__':
    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = True
        env_params = dict(robot_name="allegro_hand_xarm7",object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=True,
                          no_rgb=True)

        # If a computing device is provided, designate the rendering device.
        # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
        # based on "CUDA_VISIBLE_DEVICES".
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = AllegroRelocateRLEnv(**env_params)

        # Create camera
        environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

        # Specify observation modality
        environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
        return environment


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    print("For state task, observation is a numpy array. For visual tasks, observation is a python dict.")

    print("Observation keys")
    print(obs.keys())

    simple_pc = SimplePointCloud()


    tic = time()
    rl_steps = 1000

    for _ in range(rl_steps):
        action = np.zeros(env.action_space.shape)
        action[0] = 0.002  # Moving forward ee link in x-axis
        obs, reward, done, info = env.step(action)
        env.render()

        simple_pc.render(obs)
    elapsed_time = time() - tic




    # pc = obs["relocate-point_cloud"]
    # The name of the key in observation is "CAMERA_NAME"-"MODALITY_NAME".
    # While CAMERA_NAME is defined in task_setting.CAMERA_CONFIG["relocate"], name is point_cloud.
    # See example_use_multi_camera_visual_env.py for more modalities.

    simulation_steps = rl_steps * env.frame_skip
    print(f"Single process for point-cloud environment with {rl_steps} RL steps "
          f"(= {simulation_steps} simulation steps) takes {elapsed_time}s.")
    print("Keep in mind that using multiple processes during RL training can significantly increase the speed.")
    env.scene = None

    # # Note1: point cloud are represented x-forward convention in robot frame, see visualization to get a sense of frame
    # # Note2: you may also need to remove the points with smaller depth
    # # Note3: as described in the paper, the point are cropped into the robot workspace without background and table
    # cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([cloud, coordinate])
