from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from dexpoint2.env.rl_env.base_double import BaseDoubleRLEnv
from dexpoint2.env.sim_env.relocate_env_double import DoubleLabRelocateEnv
from dexpoint2.real_world import lab

OBJECT_LIFT_LOWER_LIMIT = -0.03


class DoubleAllegroRelocateRLEnv(DoubleLabRelocateEnv, BaseDoubleRLEnv):
    def __init__(self, use_gui=False, frame_skip=10, robot_name="allegro_hand_xarm7",
                 rotation_reward_weight=0, object_category="YCB", object_name="tomato_soup_can",
                 randomness_scale=1, friction=1, root_frame="world", **renderer_kwargs):
        if "allegro" not in robot_name or "free" in robot_name:
            raise ValueError(f"Robot name: {robot_name} is not valid xarm allegro robot.")

        super().__init__(use_gui, frame_skip, object_category, object_name, randomness_scale, friction,
                         **renderer_kwargs)

        # Base class
        self.setup()
        self.env_name='double_arm'
        self.rotation_reward_weight = rotation_reward_weight

        # Parse link name
        self.l_palm_link_name = self.l_robot_info.palm_name
        self.l_palm_link = [link for link in self.l_robot.get_links() if link.get_name() == self.l_palm_link_name][0]
        self.r_palm_link_name = self.r_robot_info.palm_name
        self.r_palm_link = [link for link in self.r_robot.get_links() if link.get_name() == self.r_palm_link_name][0]

        # Base frame for observation
        self.root_frame = root_frame
        self.base_frame_pos = np.zeros(3)


        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
        finger_contact_link_name = [
            "link_15.0_tip", "link_15.0", "link_14.0",
            "link_3.0_tip", "link_3.0", "link_2.0", "link_1.0",
            "link_7.0_tip", "link_7.0", "link_6.0", "link_5.0",
            "link_11.0_tip", "link_11.0", "link_10.0", "link_9.0"
        ]
        #left
        l_robot_link_names = [link.get_name() for link in self.l_robot.get_links()]
        self.l_finger_tip_links = [self.l_robot.get_links()[l_robot_link_names.index(name)] for name in finger_tip_names]
        self.l_finger_contact_links = [self.l_robot.get_links()[l_robot_link_names.index(name)] for name in
                                     finger_contact_link_name]
        self.l_finger_contact_ids = np.array([0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4])
        self.l_finger_tip_pos = np.zeros([len(finger_tip_names), 3])
        #right
        r_robot_link_names = [link.get_name() for link in self.r_robot.get_links()]
        self.r_finger_tip_links = [self.r_robot.get_links()[r_robot_link_names.index(name)] for name in finger_tip_names]
        self.r_finger_contact_links = [self.r_robot.get_links()[r_robot_link_names.index(name)] for name in
                                     finger_contact_link_name]
        self.r_finger_contact_ids = np.array([0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4])
        self.r_finger_tip_pos = np.zeros([len(finger_tip_names), 3])

        # self.finger_reward_scale = np.ones(len(self.l_finger_tip_links)) * 0.01
        # self.finger_reward_scale[0] = 0.04
        self.finger_reward_scale = np.ones(len(self.l_finger_tip_links)) * 0.02

        # Object, palm, target pose
        self.object_pose = self.manipulated_object.get_pose()
        self.l_palm_pose = self.l_palm_link.get_pose()
        self.l_palm_pos_in_base = np.zeros(3)
        self.l_object_in_tip = np.zeros([len(finger_tip_names), 3])

        self.r_palm_pose = self.r_palm_link.get_pose()
        self.r_palm_pos_in_base = np.zeros(3)
        self.r_object_in_tip = np.zeros([len(finger_tip_names), 3])

        self.target_in_object = np.zeros([3])
        self.target_in_object_angle = np.zeros([1])
        self.object_lift = 0

        # Contact buffer
        self.l_robot_object_contact = np.zeros(len(finger_tip_names) + 1)  # four tip, palm 
        self.r_robot_object_contact = np.zeros(len(finger_tip_names) + 1)  # four tip, palm 

        
        self.reward_info = {
            "reward_l_finger_object_dis": 0.0,
            "reward_l_palm_object_dis": 0.0,
            "reward_l_contact": 0.0,
            "reward_r_finger_object_dis": 0.0,
            "reward_r_palm_object_dis": 0.0,
            "reward_r_contact": 0.0,
            "reward_lift": 0.0,
            "reward_target_obj_dis": 0.0,
            "reward_other": 0.0,
        }
        self.clip_network_output = True
        self.print_is = False


    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # Gym reset function
        if seed is not None:
            self.seed(seed)

        self.reset_internal()
        # Set robot qpos
        l_qpos = np.zeros(self.l_robot.dof)
        l_xarm_qpos = self.l_robot_info.arm_init_qpos
        l_qpos[:self.arm_dof] = l_xarm_qpos
        self.l_robot.set_qpos(l_qpos)
        self.l_robot.set_drive_target(l_qpos)

        r_qpos = np.zeros(self.r_robot.dof)
        r_xarm_qpos = self.r_robot_info.arm_init_qpos
        r_qpos[:self.arm_dof] = r_xarm_qpos
        self.r_robot.set_qpos(r_qpos)
        self.r_robot.set_drive_target(r_qpos)
        

        # Set robot pose
        l_init_pos = np.array(lab.l_ROBOT2BASE.p) + self.l_robot_info.root_offset
        l_init_pose = sapien.Pose(l_init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        self.l_robot.set_pose(l_init_pose)

        r_init_pos = np.array(lab.r_ROBOT2BASE.p) + self.r_robot_info.root_offset
        r_init_pose = sapien.Pose(r_init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        self.r_robot.set_pose(r_init_pose)

        #TODO set the root frame
        if self.root_frame == "robot":
            self.base_frame_pos = self.l_robot.get_pose().p
            
        elif self.root_frame == "world":
            self.base_frame_pos = np.zeros(3)
            
        else:
            raise NotImplementedError
        self.update_cached_state()
        self.update_imagination(reset_goal=True)
        return self.get_observation()
    
    def update_cached_state(self):
        for i, link in enumerate(self.l_finger_tip_links):
            self.l_finger_tip_pos[i] = self.l_finger_tip_links[i].get_pose().p
        l_check_contact_links = self.l_finger_contact_links + [self.l_palm_link]
        l_contact_boolean = self.check_actor_pair_contacts(l_check_contact_links, self.manipulated_object)
        self.l_robot_object_contact[:] = np.clip(np.bincount(self.l_finger_contact_ids, weights=l_contact_boolean), 0, 1)

        for i, link in enumerate(self.r_finger_tip_links):
            self.r_finger_tip_pos[i] = self.r_finger_tip_links[i].get_pose().p
        r_check_contact_links = self.r_finger_contact_links + [self.r_palm_link]
        r_contact_boolean = self.check_actor_pair_contacts(r_check_contact_links, self.manipulated_object)
        self.r_robot_object_contact[:] = np.clip(np.bincount(self.r_finger_contact_ids, weights=r_contact_boolean), 0, 1)

        self.object_pose = self.manipulated_object.get_pose()

        self.l_palm_pose = self.l_palm_link.get_pose()
        self.l_palm_pos_in_base = self.l_palm_pose.p - self.base_frame_pos
        self.l_palm_in_object = self.l_palm_link.get_pose().p - self.object_pose.p
        self.l_object_in_tip = self.object_pose.p[None, :] - self.l_finger_tip_pos

        self.r_palm_pose = self.r_palm_link.get_pose()
        self.r_palm_pos_in_base = self.r_palm_pose.p - self.base_frame_pos
        self.r_palm_in_object = self.r_palm_link.get_pose().p - self.object_pose.p
        self.r_object_in_tip = self.object_pose.p[None, :] - self.r_finger_tip_pos

        self.object_lift = self.object_pose.p[2] - self.object_height
        self.target_in_object = self.target_pose.p - self.object_pose.p
        self.target_in_object_angle[0] = np.arccos(
            np.clip(np.power(np.sum(self.object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
        self.l_is_contact = np.sum(self.l_robot_object_contact) >= 2
        self.r_is_contact = np.sum(self.r_robot_object_contact) >= 2


    def get_oracle_state(self):
        object_pos = self.object_pose.p
        object_quat = self.object_pose.q
        object_pose_vec = np.concatenate([object_pos - self.base_frame_pos, object_quat])
        l_robot_qpos_vec = self.l_robot.get_qpos()
        r_robot_qpos_vec = self.r_robot.get_qpos()


        oracle_state=np.concatenate([
            l_robot_qpos_vec, r_robot_qpos_vec, self.l_palm_pos_in_base, self.r_palm_pos_in_base,  # （dof + 3)*2
            object_pose_vec, self.l_object_in_tip.flatten(), self.r_object_in_tip.flatten(),self.l_robot_object_contact,self.r_robot_object_contact,  # 7 + (12 + 5)*2
            self.target_in_object, self.target_pose.q, self.target_in_object_angle  # 3 + 4 + 1
        ], dtype=np.float32)
        return oracle_state

    def get_robot_state(self):
        l_robot_qpos_vec = self.l_robot.get_qpos()
        r_robot_qpos_vec = self.r_robot.get_qpos()

        robot_state=np.concatenate([
            l_robot_qpos_vec,r_robot_qpos_vec, self.l_palm_pos_in_base,self.r_palm_pos_in_base,
            self.target_pose.p - self.base_frame_pos, self.target_pose.q
        ], dtype=np.float32)
        return robot_state

    def get_info(self):
        # 将reward信息返回
        info = {
            "reward_l_finger_object_dis": self.reward_l_finger_object_dis_val,
            "reward_l_palm_object_dis": self.reward_l_palm_object_dis_val,
            "reward_l_contact": self.reward_l_contact_val,
            "reward_r_finger_object_dis": self.reward_r_finger_object_dis_val,
            "reward_r_palm_object_dis": self.reward_r_palm_object_dis_val,
            "reward_r_contact": self.reward_r_contact_val,
            "reward_lift": self.reward_lift_val,
            "reward_target_obj_dis": self.reward_target_obj_dis_val,
            "reward_other": self.reward_other_val,
        }
        if self.print_is:
            print(info)
        return info

    def get_reward(self, action):
        reward = 0.0
        reward += self.reward_l_palm_object_dis(action)
        reward += self.reward_l_finger_object_dis(action)
        reward += self.reward_l_contact(action)
        reward += self.reward_r_palm_object_dis(action)
        reward += self.reward_r_finger_object_dis(action)
        reward += self.reward_r_contact(action)
        reward += self.reward_lift(action)
        reward += self.reward_target_obj_dis(action)
        reward += self.reward_other(action)

        return reward / 10
    
    def reward_l_palm_object_dis(self, action):
        reward = 0.0
        if np.sum(self.l_robot_object_contact) < 10 :
        # if True:
            l_palm_in_object = np.linalg.norm(self.l_palm_in_object)
            self.l_palm_in_object_val = np.clip(l_palm_in_object, 0.07, 0.8) #0.13
            reward = np.sum(1.0 / (0.01 + self.l_palm_in_object_val) * 0.05)
        self.reward_l_palm_object_dis_val = reward
        return reward
    
    def reward_r_palm_object_dis(self, action):
        reward = 0.0
        if np.sum(self.r_robot_object_contact) < 10 :
        # if True:
            r_palm_in_object = np.linalg.norm(self.r_palm_in_object)
            self.r_palm_in_object_val = np.clip(r_palm_in_object, 0.07, 0.8)
            reward = np.sum(1.0 / (0.01 + self.r_palm_in_object_val) * 0.05)
        self.reward_r_palm_object_dis_val = reward
        return reward

    def reward_l_finger_object_dis(self, action):
        reward = 0.0
        if np.sum(self.l_robot_object_contact) < 10 :
            if self.l_palm_in_object_val < 0.10:
            # if True:
                l_finger_object_dist = np.linalg.norm(self.l_object_in_tip, axis=1, keepdims=False)
                l_finger_object_dist = np.clip(l_finger_object_dist, 0.05, 0.8) #0.05 0.15
                reward = np.sum(1.0 / (0.01 + l_finger_object_dist) * self.finger_reward_scale)
        self.reward_l_finger_object_dis_val = reward
        return reward
    
    def reward_r_finger_object_dis(self, action):
        reward = 0.0
        if np.sum(self.r_robot_object_contact) < 10 :
            if self.r_palm_in_object_val < 0.10:
            # if True:
                r_finger_object_dist = np.linalg.norm(self.r_object_in_tip, axis=1, keepdims=False)
                r_finger_object_dist = np.clip(r_finger_object_dist, 0.05, 0.8)
                reward = np.sum(1.0 / (0.01 + r_finger_object_dist) * self.finger_reward_scale)
        self.reward_r_finger_object_dis_val = reward
        return reward
    
    def reward_l_contact(self, action):
        reward = 0.0
        if np.sum(self.l_robot_object_contact) >= 1:
            reward += 0.25 * np.clip(np.sum(self.l_robot_object_contact), 0, 3)
        reward = reward * 2
        self.reward_l_contact_val = reward
        return reward
    
    def reward_r_contact(self, action):
        reward = 0.0
        if np.sum(self.r_robot_object_contact) >= 1:
            reward += 0.25 * np.clip(np.sum(self.r_robot_object_contact), 0, 3)
        reward = reward * 2
        self.reward_r_contact_val = reward
        return reward
    

    def reward_lift(self, action):
        self.lift = False
        reward=0
        if self.l_is_contact and self.r_is_contact:
            # reward += 0.5
            lift = np.clip(self.object_lift, 0, 0.2)
            reward += 10 * lift
            if lift > 0.02:
                self.lift = True
        reward = reward * 4
        self.reward_lift_val = reward
        return reward
    
    def reward_target_obj_dis(self, action):
        reward = 0.0
        if self.lift:
            reward = 1
            target_obj_dist = np.linalg.norm(self.target_in_object)
            reward += 1.0 / (0.04 + target_obj_dist)
        reward = reward * 4
        self.reward_target_obj_dis_val = reward
        return reward
    
    def reward_other(self, action):
        reward = 0.0
        l_qvel = self.l_robot.get_qvel()
        r_qvel = self.r_robot.get_qvel()
        #print('robot vel:', qvel.shape)
        reward += np.sum(np.clip(l_qvel, -1, 1) ** 2) * -0.01
        reward += np.sum(np.clip(r_qvel, -1, 1) ** 2) * -0.01
        reward += (self.l_cartesian_error ** 2) * -1e3
        reward += (self.r_cartesian_error ** 2) * -1e3
        self.reward_other_val = reward
        return reward




    def is_done(self):
        return bool(self.object_lift < OBJECT_LIFT_LOWER_LIMIT)

    @cached_property
    def horizon(self):
        return 200


def main_env():
    import sys, os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    
    
    from time import time
    env = DoubleAllegroRelocateRLEnv(use_gui=True, robot_name="allegro_hand_xarm7",
                               object_name="any_train", object_category="02876657", frame_skip=10,
                               use_visual_obs=True)
    base_env = env
    from dexpoint2.real_world import task_setting

    # Create camera
    env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate_double"])

    # Specify observation
    env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise_double"])
    # Specify imagination
    env.setup_imagination_config(task_setting.IMG_CONFIG["relocate_robot_only"])

    # It will check your custom environment and output additional warnings if needed
    print('obs space:', env.observation_space)
    #robot_dof = env.robot.dof
    env.seed(0)
    env.reset()



    tic = time()
    env.reset()
    tac = time()
    print(f"Reset time: {(tac - tic) * 1000} ms")

    tic = time()
    for i in range(1000):
        action = np.random.rand(44) * 2 - 1
        action[2] = 0.1
        #obs, reward, done, _ = env.step(action)
    tac = time()
    print(f"Step time: {(tac - tic)} ms")

    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer

    

    env.reset()
    # env.reset_env()
    viewer.toggle_pause(True)
    pose = env.l_palm_link.get_pose()
    for i in range(5000):
        action = np.zeros(44)
        # action[22] =-0.03
        # action[23]=0.04
        obs, reward, done, _ = env.step(action)
        env.render()
        if i == 100:
            pose_error = pose.inv() * env.l_palm_link.get_pose()
            print(pose_error)

    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
