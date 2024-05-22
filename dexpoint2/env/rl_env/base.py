from abc import abstractmethod
from functools import cached_property
from typing import Dict, Optional, Callable, List, Union, Tuple

import gym
import numpy as np
import sapien.core as sapien
import transforms3d

from dexpoint2.env.sim_env.base import BaseSimulationEnv
from dexpoint2.env.sim_env.constructor import add_default_scene_light
from dexpoint2.kinematics.kinematics_helper import PartialKinematicModel
from dexpoint2.utils.common_robot_utils import load_robot, generate_arm_robot_hand_info, \
    generate_free_robot_hand_info, FreeRobotInfo, ArmRobotInfo
from dexpoint2.utils.random_utils import np_random
from EigenGrasp.eigengrasp import EigenGrasp



VISUAL_OBS_RETURN_TORCH = False
MAX_DEPTH_RANGE = 2.5
gl2sapien = sapien.Pose(q=np.array([0.5, 0.5, -0.5, -0.5]))


def recover_action(action, limit):
    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    return action


class BaseRLEnv(BaseSimulationEnv, gym.Env):
    def __init__(self, use_gui=True, frame_skip=5, use_visual_obs=False, **renderer_kwargs):
        # Do not write any meaningful in this __init__ function other than type definition,
        # Since multiple parents are presented for the child RLEnv class
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs, **renderer_kwargs)

        # Visual staff for offscreen rendering
        self.camera_infos: Dict[str, Dict] = {}
        self.camera_pose_noise: Dict[
            str, Tuple[Optional[float], sapien.Pose]] = {}  # tuple for noise level and original camera pose
        self.imagination_infos: Dict[str, float] = {}
        self.imagination_data: Dict[str, Dict[str, Tuple[sapien.ActorBase, np.ndarray, int]]] = {}
        self.imaginations: Dict[str, np.ndarray] = {}

        # RL related attributes
        self.is_robot_free: Optional[bool] = None
        self.arm_dof: Optional[int] = None
        self.rl_step: Optional[Callable] = None
        self.get_observation: Optional[Callable] = None
        self.robot_collision_links: Optional[List[sapien.Actor]] = None
        self.robot_info: Optional[Union[ArmRobotInfo, FreeRobotInfo]] = None
        self.velocity_limit: Optional[np.ndarray] = None
        self.kinematic_model: Optional[PartialKinematicModel] = None

        # Robot cache
        self.control_time_step = None
        self.ee_link_name = None
        self.ee_link: Optional[sapien.Actor] = None
        self.cartesian_error = None

        self.executor = EigenGrasp(16,2).load_from_file("EigenGrasp/grasp_model_2.pkl")
        

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, action):
        pass

    def get_info(self):
        return {}

    def update_cached_state(self):
        return

    @abstractmethod
    def is_done(self):
        pass

    @property
    def action_dim(self):
        # return self.robot.dof
        return 22

    @property
    @abstractmethod
    def horizon(self):
        return 0

    def setup(self, robot_name):
        self.robot_name = robot_name
        self.robot = load_robot(self.scene, robot_name, disable_self_collision=False)
        self.robot.set_pose(sapien.Pose(np.array([0, 0, -5])))

        info = generate_arm_robot_hand_info()[robot_name]
        self.arm_dof = info.arm_dof
        hand_dof = info.hand_dof
        limit = 0.1
        limit_hand = 1
        velocity_limit = np.array([limit] * 3 + [limit] * 3 + [np.pi*limit_hand] * hand_dof)
        self.velocity_limit = np.stack([-velocity_limit, velocity_limit], axis=1)
        start_joint_name = self.robot.get_joints()[1].get_name()
        end_joint_name = self.robot.get_active_joints()[self.arm_dof - 1].get_name()
        self.kinematic_model = PartialKinematicModel(self.robot, start_joint_name, end_joint_name)
        self.ee_link_name = self.kinematic_model.end_link_name
        self.ee_link = [link for link in self.robot.get_links() if link.get_name() == self.ee_link_name][0]

        self.robot_info = info
        self.robot_collision_links = [link for link in self.robot.get_links() if len(link.get_collision_shapes()) > 0]
        self.control_time_step = self.scene.get_timestep() * self.frame_skip
        print("frame_skip", self.frame_skip)
        print("control_time_step", self.control_time_step)

        # Choose different step function
        self.rl_step = self.arm_sim_step

        # Scene light and obs
        if self.use_visual_obs:
            self.get_observation = self.get_visual_observation
            if not self.no_rgb:
                add_default_scene_light(self.scene, self.renderer)
        else:
            self.get_observation = self.get_oracle_state

    def arm_sim_step(self, action: np.ndarray):
        current_qpos = self.robot.get_qpos()
        #print("current_qpos", current_qpos)
        ee_link_last_pose = self.ee_link.get_pose()
        # print("ee_link_last_pose", ee_link_last_pose)
        action[:6] = np.clip(action[:6], -1, 1)
        #action[6:] = np.clip(action[6:], -1, 1)
        #hand_action_res = [-0.0,-0.78539815,-0.78539815,-0.78539815,-0.0,-0.78539815,-0.78539815 ,-0.78539815 , -0.0,-0.78539815,-0.78539815,-0.78539815,0.2,-0.78539815,-0.0,-0.78539815]
        # for i in range(16):
        #      action[i+6] *= 2
        #     #action[i+6] += hand_action_res[i]
        
        # print("action", action)
        # action = np.clip(action, -1, 1)
        target_root_velocity = recover_action(action[:6], self.velocity_limit[:6])
        # print("target_root_velocity", target_root_velocity)
        palm_jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(current_qpos[:self.arm_dof])
        arm_qvel = compute_inverse_kinematics(target_root_velocity, palm_jacobian)[:self.arm_dof]
        arm_qvel = np.clip(arm_qvel, -np.pi / 1, np.pi / 1)
        # print("arm_qvel", arm_qvel)
        arm_qpos = arm_qvel * self.control_time_step + self.robot.get_qpos()[:self.arm_dof]
        # print("arm_qpos", arm_qpos)

        # hand_qpos = recover_action(action[6:], self.robot.get_qlimits()[self.arm_dof:])
        #hand_qpos = np.clip(action[6:], self.robot.get_qlimits()[self.arm_dof:][:, 0], self.robot.get_qlimits()[self.arm_dof:][:, 1])
        hand_qpos=action[6:]
        #hand_qpos= self.executor.compute_grasp(action[6:])
        # print("hand_qpos1", hand_qpos)
        # TODO 速度限制
        # allowed_hand_motion = self.velocity_limit[6:] * self.control_time_step
        # hand_qpos = np.clip(hand_qpos, current_qpos[self.arm_dof:] + allowed_hand_motion[:, 0],
        #                      current_qpos[self.arm_dof:] + allowed_hand_motion[:, 1])
        # print("hand_qpos2", hand_qpos)
        target_qpos = np.concatenate([arm_qpos, hand_qpos])
        target_qvel = np.zeros_like(target_qpos)
        target_qvel[:self.arm_dof] = arm_qvel
        self.robot.set_drive_target(target_qpos)
        self.robot.set_drive_velocity_target(target_qvel)

        for i in range(self.frame_skip):
            self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
            self.scene.step()
        self.current_step += 1

        ee_link_new_pose = self.ee_link.get_pose()
        relative_pos = ee_link_new_pose.p - ee_link_last_pose.p
        self.cartesian_error = np.linalg.norm(relative_pos - target_root_velocity[:3] * self.control_time_step)

    def reset_internal(self):
        self.current_step = 0
        if self.init_state is not None:
            self.scene.unpack(self.init_state)
        self.reset_env()
        if self.init_state is None:
            self.init_state = self.scene.pack()

        # Reset camera pose
        for cam_name, (noise_level, original_pose) in self.camera_pose_noise.items():
            if noise_level is None:
                continue
            pos_noise = self.np_random.randn(3) * noise_level * 0.03
            rot_noise = self.np_random.randn(3) * noise_level * 0.1
            quat_noise = transforms3d.euler.euler2quat(*rot_noise)
            perturb_pose = sapien.Pose(pos_noise, quat_noise)
            self.cameras[cam_name].set_local_pose(original_pose * perturb_pose)

    def step(self, action: np.ndarray):
        self.rl_step(action)
        self.update_cached_state()
        self.update_imagination(reset_goal=False)
        obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()
        # Reference: https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
        # Need to consider that is_done and timelimit can happen at the same time
        if self.current_step >= self.horizon:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info

    def setup_visual_obs_config(self, config: Dict[str, Dict]):
        for name, camera_cfg in config.items():
            if name not in self.cameras.keys():
                raise ValueError(
                    f"Camera {name} not created. Existing {len(self.cameras)} cameras: {self.cameras.keys()}")
            self.camera_infos[name] = {}
            banned_modality_set = {"point_cloud", "depth"}
            if len(banned_modality_set.intersection(set(camera_cfg.keys()))) == len(banned_modality_set):
                raise RuntimeError(f"Request both point_cloud and depth for same camera is not allowed. "
                                   f"Point cloud contains all information required by the depth.")

            # Add perturb for camera pose
            cam = self.cameras[name]
            if "pose_perturb_level" in camera_cfg:
                cam_pose_perturb = camera_cfg.pop("pose_perturb_level")
            else:
                cam_pose_perturb = None
            self.camera_pose_noise[name] = (cam_pose_perturb, cam.get_pose())

            for modality, cfg in camera_cfg.items():
                if modality == "point_cloud":
                    if "process_fn" not in cfg or "num_points" not in cfg:
                        raise RuntimeError(f"Missing process_fn or num_points in camera {name} point_cloud config.")

                self.camera_infos[name][modality] = cfg

        modality = []
        for camera_cfg in config.values():
            modality.extend(camera_cfg.keys())
        modality_set = set(modality)
        if "rgb" in modality_set and self.no_rgb:
            raise RuntimeError(f"Only point cloud, depth, and segmentation are allowed when no_rgb is enabled.")

    def setup_imagination_config(self, config: Dict[str, Dict[str, int]]):
        from dexpoint2.utils.render_scene_utils import actor_to_open3d_mesh
        import open3d as o3d
        # Imagination can only be used with point cloud representation
        for name, camera_cfg in self.camera_infos.items():
            assert "point_cloud" in camera_cfg

        acceptable_imagination = ["robot", "goal", "contact"]
        # Imagination class: 0 (observed), 1 (robot), 2 (goal), 3 (contact)
        img_dict = {}

        collision_link_names = [link.get_name() for link in self.robot_collision_links]
        for img_type, link_config in config.items():
            if img_type not in acceptable_imagination:
                raise ValueError(f"Unknown Imagination config name: {img_type}.")
            if img_type == "robot":
                img_dict["robot"] = {}
                for link_name, point_size in link_config.items():
                    if link_name not in collision_link_names:
                        raise ValueError(f"Link name {link_name} does not have collision geometry.")
                    link = [link for link in self.robot_collision_links if link.get_name() == link_name][0]
                    o3d_mesh = actor_to_open3d_mesh(link, use_collision_mesh=False, use_actor_pose=False)
                    sampled_cloud = o3d_mesh.sample_points_uniformly(number_of_points=point_size)
                    cloud_points = np.asarray(sampled_cloud.points)
                    img_dict["robot"][link_name] = (link, cloud_points, 1)
            elif img_type == "goal":
                img_dict["goal"] = {}
                # We do not use goal actor pointer to index pose. During reset, the goal actor may be removed.
                # Thus use goal actor here to fetch pose will lead to segfault
                # Instead, the attribute name is saved, so we can always find the latest goal actor
                for attr_name, point_size in link_config.items():
                    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    sampled_cloud = goal_sphere.sample_points_uniformly(number_of_points=point_size)
                    cloud_points = np.asarray(sampled_cloud.points)
                    img_dict["goal"][attr_name] = (attr_name, cloud_points, 2)
            else:
                raise NotImplementedError

        self.imagination_infos = config
        self.imagination_data = img_dict

    def update_imagination(self, reset_goal=False):
        for img_type, img_config in self.imagination_data.items():
            if img_type == "goal":
                if reset_goal:
                    imagination_goal = []
                    for link_name, (attr_name, points, img_class) in img_config.items():
                        pose = self.robot.get_pose().inv() * self.__dict__[attr_name].get_pose()
                        mat = pose.to_transformation_matrix()
                        transformed_points = points @ mat[:3, :3].T + mat[:3, 3][None, :]
                        imagination_goal.append(transformed_points)
                    self.imaginations["imagination_goal"] = np.concatenate(imagination_goal, axis=0)

            if img_type == "robot":
                imagination_robot = []
                for link_name, (actor, points, img_class) in img_config.items():
                    pose = self.robot.get_pose().inv() * actor.get_pose()
                    mat = pose.to_transformation_matrix()
                    transformed_points = points @ mat[:3, :3].T + mat[:3, 3][None, :]
                    imagination_robot.append(transformed_points)
                    self.imaginations["imagination_robot"] = np.concatenate(imagination_robot, axis=0)
                    # TODO
                    # self.imaginations["imagination_robot"] = np.zeros((96, 3))

    @property
    def obs_dim(self):
        if not self.use_visual_obs:
            return len(self.get_oracle_state())
        else:
            return len(self.get_robot_state())

    def get_robot_state(self):
        raise NotImplementedError

    def get_oracle_state(self):
        raise NotImplementedError

    def get_visual_observation(self):
        camera_obs = self.get_camera_obs()
        robot_obs = self.get_robot_state()
        oracle_obs = self.get_oracle_state()
        camera_obs.update(dict(state=robot_obs, oracle_state=oracle_obs))
        return camera_obs

    def get_camera_obs(self):
        self.scene.update_render()
        obs_dict = {}
        for name, camera_cfg in self.camera_infos.items():
            cam = self.cameras[name]
            modalities = list(camera_cfg.keys())
            texture_names = []
            for modality in modalities:
                if modality == "rgb":
                    texture_names.append("Color")
                elif modality == "depth":
                    texture_names.append("Position")
                elif modality == "point_cloud":
                    texture_names.append("Position")
                elif modality == "segmentation":
                    texture_names.append("Segmentation")
                else:
                    raise ValueError(f"Visual modality {modality} not supported.")

            await_dl_list = cam.take_picture_and_get_dl_tensors_async(texture_names)
            dl_list = await_dl_list.wait()

            for i, modality in enumerate(modalities):
                key_name = f"{name}-{modality}"
                dl_tensor = dl_list[i]
                shape = sapien.dlpack.dl_shape(dl_tensor)
                if modality in ["segmentation"]:
                    # TODO: add uint8 async
                    import torch
                    output_array = torch.from_dlpack(dl_tensor).cpu().numpy()
                else:
                    output_array = np.zeros(shape, dtype=np.float32)
                    sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dl_tensor, output_array)
                    sapien.dlpack.dl_cuda_sync()
                if modality == "rgb":
                    obs = output_array[..., :3]
                elif modality == "depth":
                    obs = -output_array[..., 2:3]
                    obs[obs[..., 0] > MAX_DEPTH_RANGE] = 0  # Set depth out of range to be 0
                elif modality == "point_cloud":
                    obs = np.reshape(output_array[..., :3], (-1, 3))
                    camera_pose = self.get_camera_to_robot_pose(name)
                    kwargs = camera_cfg["point_cloud"].get("process_fn_kwargs", {})
                    obs = camera_cfg["point_cloud"]["process_fn"](obs, camera_pose,
                                                                  camera_cfg["point_cloud"]["num_points"],
                                                                  self.np_random, **kwargs)
                    if "additional_process_fn" in camera_cfg["point_cloud"]:
                        for fn in camera_cfg["point_cloud"]["additional_process_fn"]:
                            obs = fn(obs, self.np_random)
                elif modality == "segmentation":
                    obs = output_array[..., :2].astype(np.uint8)
                else:
                    raise RuntimeError("What happen? you should not see this error!")
                obs_dict[key_name] = obs

        if len(self.imaginations) > 0:
            obs_dict.update(self.imaginations)

        return obs_dict

    def get_camera_to_robot_pose(self, camera_name):
        gl_pose = self.cameras[camera_name].get_pose()
        camera_pose = gl_pose * gl2sapien
        camera2robot = self.robot.get_pose().inv() * camera_pose
        return camera2robot.to_transformation_matrix()

    @cached_property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    @cached_property
    def observation_space(self):
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        state_space = gym.spaces.Box(low=low, high=high)
        if not self.use_visual_obs:
            return state_space
        else:
            oracle_dim = len(self.get_oracle_state())
            oracle_space = gym.spaces.Box(low=-np.inf * np.ones(oracle_dim), high=np.inf * np.ones(oracle_dim))
            obs_dict = {"state": state_space, "oracle_state": oracle_space}
            for cam_name, cam_cfg in self.camera_infos.items():
                cam = self.cameras[cam_name]
                resolution = (cam.height, cam.width)
                for modality_name in cam_cfg.keys():
                    key_name = f"{cam_name}-{modality_name}"
                    if modality_name == "rgb":
                        spec = gym.spaces.Box(low=0, high=1, shape=resolution + (3,))
                    elif modality_name == "depth":
                        spec = gym.spaces.Box(low=0, high=MAX_DEPTH_RANGE, shape=resolution + (1,))
                    elif modality_name == "point_cloud":
                        spec = gym.spaces.Box(low=-np.inf, high=np.inf,
                                              shape=(cam_cfg[modality_name]["num_points"],) + (3,))
                    elif modality_name == "segmentation":
                        spec = gym.spaces.Box(low=0, high=255, shape=resolution + (2,), dtype=np.uint8)
                    else:
                        raise RuntimeError("What happen? you should not see this error!")
                    obs_dict[key_name] = spec

            if len(self.imagination_infos) > 0:
                self.update_imagination(reset_goal=True)
                for img_name, points in self.imaginations.items():
                    num_points = points.shape[0]
                    obs_dict[img_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_points, 3))

            return gym.spaces.Dict(obs_dict)


def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping ** 2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ \
                 np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos
