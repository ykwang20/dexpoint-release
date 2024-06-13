import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import time

import argparse
from dexpoint2.env.rl_env.double_arm_env import DoubleAllegroRelocateRLEnv
from dexpoint2.real_world import task_setting
from stable_baselines_dexpoint2.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines_dexpoint2.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines_dexpoint2.ppo import PPO
from stable_baselines_dexpoint2.simple_callback import SimpleCallback
import torch

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
SAVE_DIR=os.path.join(BASE_DIR,'assets/checkpoints/')+time_now


def get_3d_policy_kwargs(extractor_name):
    feature_extractor_class = PointNetImaginationExtractorGP
    feature_extractor_kwargs = {"pc_key": "relocate-point_cloud", "gt_key": "instance_1-seg_gt",
                                "extractor_name": extractor_name,
                                "imagination_keys": [f'imagination_{key}' for key in task_setting.IMG_CONFIG['relocate_robot_only'].keys()],
                                "state_key": "state"}

    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }
    return policy_kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--freeze', dest='freeze', action='store_true', default=False)
    parser.add_argument('--task_name', type=str, default="laptop")
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--save_path', type=str, default=SAVE_DIR)
    parser.add_argument('--eigen_dim', type=int, default=None)
    args = parser.parse_args()

    task_name = args.task_name
    extractor_name = args.extractor_name
    seed = args.seed if args.seed >= 0 else random.randint(0, 100000)
    pretrain_path = args.pretrain_path
    horizon = 200
    env_iter = args.iter * horizon * args.n
    print(f"freeze: {args.freeze}")

    


    def create_env_fn():
        # object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        # object_name = np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = True
        object_name='any_train'
        object_category="02876657"
        # object_name="potted_meat_can"
        # object_category="YCB"
        robot_name = "xarm7_allegro_v2"
        env_params = dict(robot_name=robot_name, 
                          object_name=object_name,
                          object_category=object_category,
                          frame_skip=20, 
                          eigen_dim=args.eigen_dim,
                          rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, 
                          use_visual_obs=use_visual_obs, 
                          use_gui=False,
                          no_rgb=True)

        environment = DoubleAllegroRelocateRLEnv(**env_params)
        environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate_double"])
        environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise_double"])
        environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_robot_only"])
        return environment

    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")  # train on a list of envs.

    action_dim=str(6+args.eigen_dim)
    model = PPO("PointCloudPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * horizon,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=seed,
                policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
                min_lr=args.lr,
                max_lr=args.lr,
                adaptive_kl=0.02,
                target_kl=0.2,
                tensorboard_log="./logs/double/dex"+action_dim+"/"
                )
    #model = PPO.load("/home/yikai/dexpoint-release/assets/checkpoints/20240604-0715/model_850.zip",check_obs_space=False, env=env,tensorboard_log="./logs/double/dex"+action_dim+"/")
    print('policy net:',model.policy)
    if pretrain_path is not None:
        state_dict: OrderedDict = torch.load(pretrain_path)
        model.policy.features_extractor.extractor.load_state_dict(state_dict, strict=False)
        print("load pretrained model: ", pretrain_path)

    rollout = int(model.num_timesteps / (horizon * args.n))

    # after loading or init the model, then freeze it if needed
    if args.freeze:
        model.policy.features_extractor.extractor.eval()
        for param in model.policy.features_extractor.extractor.parameters():
            param.requires_grad = False
        print("freeze model!")
    print('save path:',args.save_path)
    model.learn(
        total_timesteps=int(env_iter),
        reset_num_timesteps=False,
        iter_start=rollout,
        tb_log_name=time_now,
        callback=SimpleCallback(model_save_freq=args.save_freq, model_save_path=args.save_path, rollout=0),
    )
