# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask

from terrain_base.terrain import Terrain
from terrain_base.config import terrain_config

from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


class HumanoidRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, save):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self.save = save
        
        # 设置目标偏离调试开关
        self.debug_goal_deviation = getattr(self.cfg.rewards, 'debug_goal_deviation', True)
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()

        if self.save:
            self.episode_data = {
                'observations': [[] for _ in range(self.num_envs)],
                'actions': [[] for _ in range(self.num_envs)],
                'rewards': [[] for _ in range(self.num_envs)],
                'height_map': [[] for _ in range(self.num_envs)],
                'privileged_obs': [[] for _ in range(self.num_envs)],
                'rigid_body_state': [[] for _ in range(self.num_envs)],
                'dof_state': [[] for _ in range(self.num_envs)]
            }
            self.current_episode_buffer = {
                'observations': [[] for _ in range(self.num_envs)],
                'actions': [[] for _ in range(self.num_envs)],
                'rewards': [[] for _ in range(self.num_envs)],
                'height_map': [[] for _ in range(self.num_envs)],
                'privileged_obs': [[] for _ in range(self.num_envs)],
                'rigid_body_state': [[] for _ in range(self.num_envs)],
                'dof_state': [[] for _ in range(self.num_envs)]
            }
        # init data save buffer
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0
        self.time_stamp = 0

        self.total_times = 0
        self.last_times = -1
        self.success_times = 0
        self.complete_times = 0.

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def get_data_stats(self):
        """get dataset information"""
        stats = {
            'total_episodes': 0,
            'total_samples': 0,
            'avg_episode_length': 0
        }
        for env_data in self.episode_data['observations']:
            stats['total_episodes'] += len(env_data)
            for ep in env_data:
                stats['total_samples'] += ep.shape[0]
        if stats['total_episodes'] > 0:
            stats['avg_episode_length'] = stats['total_samples'] / stats['total_episodes']
        return stats

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None

        if self.save:
            for env_idx in range(self.num_envs):
                self.current_episode_buffer['observations'][env_idx].append(
                    self.obs_buf[env_idx].cpu().numpy().copy())  
                self.current_episode_buffer['actions'][env_idx].append(
                    self.actions[env_idx].cpu().numpy().copy())      
                
                self.current_episode_buffer['rewards'][env_idx].append(
                    self.rew_buf[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['height_map'][env_idx].append(
                    self.measured_heights_data[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['rigid_body_state'][env_idx].append(
                    self.rigid_body_states[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['dof_state'][env_idx].append(
                    self.dof_state[env_idx].cpu().numpy().copy())  

                if self.privileged_obs_buf is not None:
                    self.current_episode_buffer['privileged_obs'][env_idx].append(
                        self.privileged_obs_buf[env_idx].cpu().numpy().copy())      

        if(self.cfg.rewards.is_play):
            if(self.total_times > 0):
                if(self.total_times > self.last_times):
                    print("total_times=",self.total_times)
                    print("success_rate=",self.success_times / self.total_times)
                    print("complete_rate=",(self.complete_times / self.total_times).cpu().numpy().copy())
                    self.last_times = self.total_times

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        
        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        
        # 更新关节速度历史
        self.dof_vel_history.append(self.dof_vel.clone())
        if len(self.dof_vel_history) > 3:  # 只保留最近3帧
            self.dof_vel_history.pop(0)
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        if(self.time_stamp ==5):
            self.last_foot_action = self.rigid_body_states[:, self.feet_indices, :]
            self.time_stamp=0
        else :
            self.time_stamp=self.time_stamp+1
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            self._draw_goals()
            # self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < 0.5

        # 新增：目标点偏离早停条件
        goal_deviation_cutoff = self._check_goal_deviation()

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff
        self.reset_buf |= goal_deviation_cutoff  # 添加目标偏离终止条件

        self.total_times += len(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.success_times += len(reach_goal_cutoff.nonzero(as_tuple=False).flatten())
        self.complete_times += (self.cur_goal_idx[self.reset_buf.nonzero(as_tuple=False).flatten()] / self.cfg.terrain.num_goals).sum()

    def _check_goal_deviation(self):
        """检查目标点偏离情况，当距离相差过大，且时间超过5s，且朝向和目标点不对的情况下，作为终结这个goals的早停条件"""
        try:
            # 获取当前机器人位置
            current_pos = self.root_states[:, :3]  # [num_envs, 3]
            
            # 获取下一个目标点位置
            next_goal_pos = self.next_target_pos_rel  # [num_envs, 3]
            
            # 计算到下一个目标点的距离
            distance_to_goal = torch.norm(next_goal_pos, dim=1)  # [num_envs]
            
            # 距离阈值：如果距离目标点太远
            distance_threshold = getattr(self.cfg.rewards, 'goal_deviation_distance_threshold', 3.0)
            distance_cutoff = distance_to_goal > distance_threshold
            
            # 计算当前朝向与目标方向的夹角
            current_heading = torch.atan2(self.base_lin_vel[:, 1], self.base_lin_vel[:, 0])
            target_heading = self.next_target_yaw  # 目标朝向
            
            # 计算朝向误差
            heading_error = torch.abs(current_heading - target_heading)
            # 处理角度环绕问题
            heading_error = torch.min(heading_error, 2 * torch.pi - heading_error)
            
            # 方向阈值：如果朝向偏离太多
            heading_threshold = getattr(self.cfg.rewards, 'goal_deviation_heading_threshold', torch.pi / 2)
            heading_cutoff = heading_error > heading_threshold
            
            # 时间条件：当前目标点已经存在超过指定时间
            time_threshold = getattr(self.cfg.rewards, 'goal_deviation_time_threshold', 5.0)
            time_cutoff = self.episode_length_buf > time_threshold
            
            # 综合所有条件：距离过大 AND 时间超过5s AND 朝向不对
            goal_deviation_cutoff = distance_cutoff & time_cutoff & heading_cutoff
            
            # 调试信息：记录终止原因（可选）
            if hasattr(self, 'debug_goal_deviation') and self.debug_goal_deviation:
                if torch.any(goal_deviation_cutoff):
                    terminated_envs = torch.where(goal_deviation_cutoff)[0]
                    for env_id in terminated_envs:
                        env_id = env_id.item()
                        print(f"Env {env_id}: Terminated due to goal deviation - Distance: {distance_to_goal[env_id]:.2f}m > {distance_threshold:.2f}m, Time: {self.episode_length_buf[env_id]:.2f}s > {time_threshold:.2f}s, Heading: {heading_error[env_id]:.2f}rad > {heading_threshold:.2f}rad")
            
            return goal_deviation_cutoff
            
        except Exception as e:
            # 如果出现任何错误，返回全False（不终止）
            print(f"Warning: Error in _check_goal_deviation: {e}")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        if self.save:
            for env_id in env_ids:
                try:
                    if len(self.current_episode_buffer['observations'][env_id]) > 750:
                        # 转换为numpy数组
                        episode_obs = np.stack(self.current_episode_buffer['observations'][env_id])  # [T,*]
                        episode_act = np.stack(self.current_episode_buffer['actions'][env_id])       # [T,*]
                        episode_rew = np.stack(self.current_episode_buffer['rewards'][env_id])      # [T]
                        episode_hei = np.stack(self.current_episode_buffer['height_map'][env_id])      # [T, 396]
                        episode_body = np.stack(self.current_episode_buffer['rigid_body_state'][env_id]) # [T,13,13] first is root
                        episode_dof = np.stack(self.current_episode_buffer['dof_state'][env_id])
                      
                        # 存入主数据存储
                        self.episode_data['observations'][env_id].append(episode_obs)
                        self.episode_data['actions'][env_id].append(episode_act)
                        self.episode_data['rewards'][env_id].append(episode_rew)
                        self.episode_data['height_map'][env_id].append(episode_hei)
                        self.episode_data['rigid_body_state'][env_id].append(episode_body)
                        self.episode_data['dof_state'][env_id].append(episode_dof)

                        
                        # 处理privileged观测
                        if self.privileged_obs_buf is not None:
                            episode_priv = np.stack(self.current_episode_buffer['privileged_obs'][env_id]) # [T,*]
                            self.episode_data['privileged_obs'][env_id].append(episode_priv)
                        
                        # 清空当前buffer
                        self.current_episode_buffer['observations'][env_id] = []
                        self.current_episode_buffer['actions'][env_id] = []
                        self.current_episode_buffer['rewards'][env_id] = []
                        self.current_episode_buffer['height_map'][env_id] = []
                        self.current_episode_buffer['privileged_obs'][env_id] = []
                        self.current_episode_buffer['rigid_body_state'][env_id] = []
                        self.current_episode_buffer['dof_state'][env_id] = []
                        
                        print(f"Env {env_id} have saved {episode_obs.shape[0]} step data")
                except Exception as e:
                    print(f"An error occured when saving env {env_id}: {str(e)}")
        
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_foot_action[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        # print("len_reward=",len(self.reward_functions))
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if name !="success_rate" or name !="complete_rate":
                self.episode_sums[name] += rew
                
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ 
        Computes observations
        即本体感知
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw
        obs_buf = torch.cat((#skill_vector, 
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3] # 3
                            imu_obs,    #[1,2]  2 只包含roll和pitch
                            0*self.delta_yaw[:, None], # 1
                            self.delta_yaw[:, None], # 1
                            self.delta_next_yaw[:, None],  # 1
                            0*self.commands[:, 0:2],  # 2
                            self.commands[:, 0:1],  #[1,1]  # 1
                            (self.env_class != 17).float()[:, None],  #1
                            (self.env_class == 17).float()[:, None], # 1
                            (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos, # h1:19
                            self.dof_vel * self.obs_scales.dof_vel,  # h1:19
                            self.action_history_buf[:, -1], # h1:19
                            self.contact_filt.float()-0.5, # 2
                            ),dim=-1)

        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                   0 * self.base_lin_vel,
                                   0 * self.base_lin_vel), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            
            self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        obs_buf[:, 6:8] = 0  

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
            
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        start = time()
        print("*"*80)
        mesh_type = terrain_config.mesh_type

        if mesh_type=='None':
            self._create_ground_plane()
        else:
            self.terrain = Terrain(self.num_envs)
            self._create_trimesh()

        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.8*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip
        
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights, self.measured_heights_data  = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip

        # set small commands to zero
        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        move_up =dis_to_origin > 0.8*threshold
        move_down = dis_to_origin < 0.4*threshold

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)

        self.dof_pos = self.dof_state[...,0]
        self.dof_vel = self.dof_state[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_history = []  # 用于计算关节加速度的历史速度
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_foot_action = torch.zeros_like(self.rigid_body_states[:, self.feet_indices, :])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        # self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points, self.height_points_data = self._init_height_points()
        self.measured_heights = 0
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            # print("rigid_body_names=",self.gym.get_actor_rigid_body_names(env_handle, actor_handle))

            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        # print("rigid_body_names=",self.gym.get_actor_rigid_body_names(env_handle, actor_handle))

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # print("DOF names:", self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            humanoid_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "Humanoid", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, humanoid_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, humanoid_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, humanoid_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(humanoid_handle)
            
            self.attach_camera(i, env_handle, humanoid_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        # print("open=",self.cfg.domain_rand.randomize_friction)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        # print("name=",feet_names)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
 
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if terrain_config.mesh_type == "None":
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
        else:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level # 2
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1#np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(255, 0, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        if self.save:
            heights = self.measured_heights_data[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points_data[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(2):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)

        # only for recording dataset, not for policy
        y_data = torch.tensor(self.cfg.terrain.dataset_points_y, device=self.device, requires_grad=False)
        x_data = torch.tensor(self.cfg.terrain.dataset_points_x, device=self.device, requires_grad=False)
        grid_x_data, grid_y_data = torch.meshgrid(x_data, y_data)
        self.num_height_points_data = grid_x_data.numel()
        points_data = torch.zeros(self.num_envs, self.num_height_points_data, 3, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]

            # visualize saved height point
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points_data,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points_data,2), device=self.device).squeeze() + offset
            points_data[i, :, 0] = grid_x_data.flatten() #+ xy_noise[:, 0]
            points_data[i, :, 1] = grid_y_data.flatten() #+ xy_noise[:, 1]
        return points, points_data

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
            points_data = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points_data), self.height_points_data[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
            points_data = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points_data), self.height_points_data) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        points_data += self.terrain.cfg.border_size
        points_data = (points_data/self.terrain.cfg.horizontal_scale).long()
        px_data = points_data[:, :, 0].view(-1)
        py_data = points_data[:, :, 1].view(-1)
        px_data = torch.clip(px_data, 0, self.height_samples.shape[0]-2)
        py_data = torch.clip(py_data, 0, self.height_samples.shape[1]-2)
        heights1_data = self.height_samples[px_data, py_data]
        heights2_data = self.height_samples[px_data+1, py_data]
        heights3_data = self.height_samples[px_data, py_data+1]
        heights_data = torch.min(heights1_data, heights2_data)
        heights_data = torch.min(heights_data, heights3_data)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale, heights_data.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)


    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_forward_progress(self):
        # 奖励前进进度：鼓励机器人向目标方向前进
        # 计算当前朝向与目标方向的夹角
        current_heading = torch.atan2(self.base_lin_vel[:, 1], self.base_lin_vel[:, 0])
        target_heading = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        heading_error = torch.abs(current_heading - target_heading)
        
        # 前进速度奖励 - 更积极地奖励运动
        forward_velocity = self.base_lin_vel[:, 0]  # x方向速度
        forward_reward = torch.clamp(forward_velocity, min=0.0)  # 只奖励正向速度
        
        # 方向对齐奖励
        direction_reward = torch.exp(-heading_error)
        
        # 添加运动鼓励：即使命令速度很小，也鼓励运动
        motion_encouragement = torch.clamp(forward_velocity, min=0.0, max=1.0)
        
        return forward_reward * direction_reward + 0.1 * motion_encouragement

    def _reward_step_length(self):
        # 奖励合适的步长：鼓励机器人迈出合适的步子
        # 计算脚部位置变化
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]  # [num_envs, num_feet, 3]
        
        # 计算步长（脚部在x方向的位置差）
        left_foot_x = foot_positions[:, 0, 0]  # 左脚x位置
        right_foot_x = foot_positions[:, 1, 0]  # 右脚x位置
        step_length = torch.abs(left_foot_x - right_foot_x)
        
        # 理想的步长范围（根据机器人尺寸调整）
        ideal_step_length = 0.3  # 理想步长
        step_length_error = torch.abs(step_length - ideal_step_length)
        
        # 奖励接近理想步长
        return torch.exp(-step_length_error)

    def _reward_foot_swing_height(self):
        # 奖励脚部摆动高度：鼓励脚部抬起足够高度避免绊倒
        foot_heights = self.rigid_body_states[:, self.feet_indices, 2]  # 脚部z坐标
        ground_height = 0.0  # 地面高度
        
        # 计算脚部离地高度
        foot_clearance = foot_heights - ground_height
        
        # 奖励脚部抬起足够高度（避免过低）
        min_clearance = 0.05  # 最小离地高度
        clearance_reward = torch.clamp(foot_clearance - min_clearance, min=0.0)
        
        return torch.sum(clearance_reward, dim=1)

    def _reward_gait_coordination(self):
        # 奖励步态协调性：鼓励左右腿交替运动
        # 检测脚部接触地面
        foot_contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        
        # 计算接触模式：如果只有一只脚接触地面，给予奖励
        num_contacts = torch.sum(foot_contacts, dim=1)
        alternating_contact = (num_contacts == 1)  # 只有一只脚接触地面
        
        # 至少有一只脚接触地面
        at_least_one_contact = torch.any(foot_contacts, dim=1)
        
        return (alternating_contact & at_least_one_contact).float()

    def _reward_motion_penalty(self):
        # 直接惩罚静止状态：当机器人静止时给予惩罚
        # 检测机器人是否静止
        base_velocity = torch.norm(self.base_lin_vel[:, :2], dim=1)  # 基座水平速度
        joint_velocity = torch.sum(torch.abs(self.dof_vel), dim=1)   # 关节速度
        
        # 如果基座速度和关节速度都很小，认为机器人静止
        is_still = (base_velocity < 0.05) & (joint_velocity < 0.1)
        
        # 静止时给予惩罚
        return -1.0 * is_still.float()

    def _reward_foot_crossing_penalty(self):
        # 惩罚脚部交叉：防止左右脚在运动时交叉
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]  # [num_envs, num_feet, 3]
        
        # 获取左右脚位置
        left_foot_pos = foot_positions[:, 0, :2]  # 左脚xy位置
        right_foot_pos = foot_positions[:, 1, :2]  # 右脚xy位置
        
        # 计算脚部在y方向的距离
        foot_y_distance = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])
        
        # 如果脚部在y方向距离太小，说明可能交叉了
        min_safe_distance = 0.1  # 最小安全距离
        crossing_penalty = torch.clamp(min_safe_distance - foot_y_distance, min=0.0)
        
        return -crossing_penalty

    def _reward_alternating_gait(self):
        # 交替步态奖励：鼓励左右腿交替运动
        # 检测脚部接触状态
        foot_contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        
        # 计算接触模式
        left_contact = foot_contacts[:, 0]  # 左脚接触
        right_contact = foot_contacts[:, 1]  # 右脚接触
        
        # 奖励交替接触：只有一只脚接触地面
        alternating = (left_contact & ~right_contact) | (~left_contact & right_contact)
        
        # 避免双脚同时离地
        at_least_one_contact = left_contact | right_contact
        
        return (alternating & at_least_one_contact).float()

    def _reward_stride_consistency(self):
        # 步幅一致性奖励：鼓励保持一致的步幅
        # 计算当前步长
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]
        left_foot_x = foot_positions[:, 0, 0]
        right_foot_x = foot_positions[:, 1, 0]
        current_stride = torch.abs(left_foot_x - right_foot_x)
        
        # 理想步幅范围
        min_stride = 0.2
        max_stride = 0.4
        
        # 奖励在理想范围内的步幅
        stride_reward = torch.where(
            (current_stride >= min_stride) & (current_stride <= max_stride),
            torch.ones_like(current_stride),
            torch.zeros_like(current_stride)
        )
        
        return stride_reward

    def _reward_body_balance(self):
        # 身体平衡奖励：鼓励保持身体平衡
        # 惩罚过大的roll和pitch角度
        orientation = self.root_states[:, 3:7]  # 四元数
        
        # 转换为欧拉角（简化计算）
        # 这里使用投影重力作为平衡指标
        balance_penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        
        # 奖励平衡状态
        return torch.exp(-2.0 * balance_penalty)

    def _extract_terrain_features(self):
        """从高度图中提取地形特征"""
        # 前方地形分析 (前方60个点，约3米范围)
        forward_heights = self.measured_heights[:, :60]  # 前方60个点
        forward_heights = forward_heights.view(self.num_envs, 6, 10)  # 6x10网格
        
        # 计算统计特征
        height_mean = torch.mean(forward_heights, dim=(1,2))
        height_std = torch.std(forward_heights, dim=(1,2))
        height_max = torch.max(forward_heights, dim=(1,2))[0]
        height_min = torch.min(forward_heights, dim=(1,2))[0]
        
        # 计算梯度特征
        height_gradient_x = torch.diff(forward_heights, dim=1)  # x方向梯度
        height_gradient_y = torch.diff(forward_heights, dim=2)  # y方向梯度
        max_gradient_x = torch.max(torch.abs(height_gradient_x), dim=(1,2))[0]
        max_gradient_y = torch.max(torch.abs(height_gradient_y), dim=(1,2))[0]
        max_gradient = torch.max(max_gradient_x, max_gradient_y)
        
        # 基于规则的地形分类
        is_flat = (height_std < 0.02).float()
        has_gap = (height_std > 0.1).float() & (max_gradient > 0.15).float()
        has_step = (height_std > 0.05).float() & (max_gradient > 0.08).float() & (height_std < 0.1).float()
        has_pit = (height_min < -0.3).float()
        
        return is_flat, has_gap, has_step, has_pit

    def _reward_terrain_adaptive(self):
        """楼梯地形适应奖励 - 检测高度差并鼓励抬腿"""
        # 前方地形分析 (前方30个点，约1.5米范围)
        forward_heights = self.measured_heights[:, :30]  # 前方30个点
        forward_heights = forward_heights.view(self.num_envs, 5, 6)  # 5x6网格
        
        # 计算前方地形的高度差
        height_diff = torch.max(forward_heights.view(self.num_envs, -1), dim=1)[0] - torch.min(forward_heights.view(self.num_envs, -1), dim=1)[0]
        
        # 检测是否有楼梯（高度差大于阈值）
        has_stairs = (height_diff > 0.10)  # 10cm以上的高度差认为是楼梯
        
        # 基础任务奖励
        task_reward = self._reward_next_goal_progress()
        
        # 楼梯特定的抬腿奖励
        stair_reward = self._reward_stair_climbing() * has_stairs.float()
        
        # 动态权重调整
        terrain_weights = torch.ones_like(height_diff)
        terrain_weights = torch.where(has_stairs, 1.5, terrain_weights)  # 楼梯地形权重1.5倍
        
        return (task_reward + stair_reward) * terrain_weights

    def _reward_normal_walking(self):
        """平地行走奖励"""
        # 鼓励稳定的行走
        forward_velocity = self.base_lin_vel[:, 0]
        forward_reward = torch.clamp(forward_velocity, min=0.0, max=1.0)
        
        # 鼓励身体平衡
        balance_reward = torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))
        
        return forward_reward * balance_reward

    def _reward_jump_preparation(self):
        """跳跃准备奖励"""
        # 鼓励蹲下准备跳跃
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        crouch_reward = torch.exp(-torch.square(base_height - 0.8))  # 鼓励蹲下到0.8米高度
        
        # 鼓励腿部弯曲
        leg_bend_reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:, [0,1,2,3,4,5]] - 0.3), dim=1))
        
        # 鼓励停止前进，准备跳跃
        stop_reward = torch.exp(-torch.square(self.base_lin_vel[:, 0]))
        
        return crouch_reward * leg_bend_reward * stop_reward

    def _reward_step_preparation(self):
        """台阶准备奖励"""
        # 鼓励抬腿
        foot_lift_reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:, [6,7,8,9,10,11]] - 0.5), dim=1))
        
        # 鼓励身体前倾
        forward_tilt = torch.atan2(self.projected_gravity[:, 0], self.projected_gravity[:, 2])
        tilt_reward = torch.exp(-torch.square(forward_tilt - 0.1))  # 鼓励轻微前倾
        
        return foot_lift_reward * tilt_reward

    def _reward_pit_avoidance(self):
        """坑洞避免奖励"""
        # 鼓励停止前进
        stop_reward = torch.exp(-torch.square(self.base_lin_vel[:, 0]))
        
        # 鼓励身体后仰
        backward_tilt = torch.atan2(self.projected_gravity[:, 0], self.projected_gravity[:, 2])
        tilt_reward = torch.exp(-torch.square(backward_tilt + 0.1))  # 鼓励轻微后仰
        
        return stop_reward * tilt_reward

    def _reward_stair_climbing(self):
        """楼梯爬行奖励 - 鼓励抬腿和身体前倾"""
        # H1机器人有12个腿部关节：左腿6个 + 右腿6个
        # 左腿关节：hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll (索引0-5)
        # 右腿关节：hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll (索引6-11)
        
        # 鼓励抬腿（主要是hip_pitch和knee关节）
        left_hip_pitch = self.dof_pos[:, 2]   # 左腿hip_pitch
        left_knee = self.dof_pos[:, 3]        # 左腿knee
        right_hip_pitch = self.dof_pos[:, 8]  # 右腿hip_pitch
        right_knee = self.dof_pos[:, 9]       # 右腿knee
        
        # 鼓励腿部弯曲（抬腿准备）
        left_lift = torch.exp(-torch.square(left_hip_pitch - 0.2)) * torch.exp(-torch.square(left_knee - 0.4))
        right_lift = torch.exp(-torch.square(right_hip_pitch - 0.2)) * torch.exp(-torch.square(right_knee - 0.4))
        foot_lift_reward = (left_lift + right_lift) / 2.0
        
        # 鼓励身体前倾（准备上楼梯）
        forward_tilt = torch.atan2(self.projected_gravity[:, 0], self.projected_gravity[:, 2])
        tilt_reward = torch.exp(-torch.square(forward_tilt - 0.15))  # 鼓励前倾15度
        
        # 鼓励减速（楼梯前减速）
        speed_control = torch.exp(-torch.square(self.base_lin_vel[:, 0]))
        
        # 鼓励身体平衡
        balance_reward = torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))
        
        return foot_lift_reward * tilt_reward * speed_control * balance_reward

    def _reward_next_goal_alignment(self):
        # 基于下一个目标点的方向对齐奖励
        # 计算当前朝向与下一个目标方向的夹角
        current_heading = torch.atan2(self.base_lin_vel[:, 1], self.base_lin_vel[:, 0])
        next_goal_heading = self.next_target_yaw  # 下一个目标朝向
        
        # 计算朝向误差
        heading_error = torch.abs(current_heading - next_goal_heading)
        # 处理角度环绕问题
        heading_error = torch.min(heading_error, 2 * torch.pi - heading_error)
        
        # 奖励朝向对齐
        return torch.exp(-2.0 * heading_error)

    def _reward_next_goal_progress(self):
        # 基于下一个目标点的前进进度奖励
        # 计算到下一个目标点的距离
        current_distance = torch.norm(self.next_target_pos_rel, dim=1)
        
        # 奖励距离减少（向目标前进）
        # 使用指数衰减，距离越近奖励越高
        progress_reward = torch.exp(-current_distance)
        
        # 添加方向性奖励：确保机器人朝向目标
        # 使用next_target_yaw和yaw计算方向对齐，避免依赖delta_next_yaw
        direction_alignment = torch.cos(self.next_target_yaw - self.yaw)
        direction_reward = torch.clamp(direction_alignment, min=0.0)  # 只奖励正向对齐
        
        return progress_reward * direction_reward

    def _reward_smoothness(self):
        """动作平滑性奖励: ||a_t - 2a_{t-1} + a_{t-2}||^2"""
        try:
            if hasattr(self, 'last_actions') and hasattr(self, 'last_last_actions'):
                smoothness_penalty = torch.sum(torch.square(self.actions - 2 * self.last_actions + self.last_last_actions), dim=1)
                return -smoothness_penalty
        except:
            pass
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_joint_acc(self):
        """关节加速度奖励: ||θ̈||^2"""
        try:
            if hasattr(self, 'dof_vel_history') and len(self.dof_vel_history) >= 2:
                # 计算关节加速度
                joint_acc = (self.dof_vel - self.dof_vel_history[-1]) / self.dt
                return -torch.sum(torch.square(joint_acc), dim=1)
        except:
            pass
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_joint_power(self):
        """关节功率奖励: |τ · θ̇^T|"""
        # 计算关节功率
        joint_power = torch.abs(torch.sum(self.torques * self.dof_vel, dim=1))
        return -joint_power

    def _reward_body_height(self):
        """身体高度奖励: (h^target - h)^2"""
        target_height = 1.0  # 目标高度
        current_height = self.root_states[:, 2]
        height_error = torch.square(target_height - current_height)
        return -height_error

    def _reward_feet_clearance(self):
        """脚部清除奖励: Σ_feet (p_z^target - p_z^i)^2 · v_xy^i"""
        target_foot_height = 0.05  # 目标脚部高度
        foot_positions = self.rigid_body_states[:, self.feet_indices, 0:3]
        foot_heights = foot_positions[:, :, 2]
        # 获取脚部速度（rigid_body_states的索引7-9是速度）
        foot_velocities = self.rigid_body_states[:, self.feet_indices, 7:10]
        foot_xy_vel = torch.norm(foot_velocities[:, :, :2], dim=2)
        
        clearance_error = torch.square(target_foot_height - foot_heights)
        clearance_reward = torch.sum(clearance_error * foot_xy_vel, dim=1)
        return -clearance_reward

    def _reward_joint_tracking_err(self):
        """关节跟踪误差奖励: Σ_all joints |θ_i - θ_i^target|^2"""
        # 使用默认关节位置作为目标
        default_positions = torch.zeros_like(self.dof_pos)
        tracking_error = torch.sum(torch.square(self.dof_pos - default_positions), dim=1)
        return -tracking_error

    def _reward_arm_joint_dev(self):
        """手臂关节偏差奖励: Σ_arm joints |θ_i - θ_i^default|^2"""
        # H1机器人只有12个关节，没有独立的手臂关节
        # 暂时返回零奖励，因为H1是腿部机器人
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_hip_joint_dev(self):
        """髋关节偏差奖励: 固定上半身并防止手脚交叉"""
        # H1机器人的髋关节索引
        # 左腿: hip_yaw(0), hip_roll(1), hip_pitch(2)
        # 右腿: hip_yaw(6), hip_roll(7), hip_pitch(8)
        
        # 1. 固定上半身：惩罚hip_roll和hip_pitch偏离默认位置
        hip_roll_pitch_indices = [1, 2, 7, 8]  # 左右腿的hip_roll和hip_pitch
        hip_roll_pitch_positions = self.dof_pos[:, hip_roll_pitch_indices]
        default_roll_pitch_positions = torch.zeros_like(hip_roll_pitch_positions)
        roll_pitch_deviation = torch.sum(torch.square(hip_roll_pitch_positions - default_roll_pitch_positions), dim=1)
        
        # 2. 防止手脚交叉：控制hip_yaw关节
        # 左腿hip_yaw(索引0)和右腿hip_yaw(索引6)应该保持对称
        left_hip_yaw = self.dof_pos[:, 0]   # 左腿hip_yaw
        right_hip_yaw = self.dof_pos[:, 6]  # 右腿hip_yaw
        
        # 计算hip_yaw的对称性：左右腿hip_yaw应该相反（对称）
        yaw_symmetry_error = torch.square(left_hip_yaw + right_hip_yaw)  # 理想情况下 left_yaw + right_yaw = 0
        
        # 3. 限制hip_yaw的范围，防止过度旋转
        yaw_range_penalty = torch.sum(torch.square(torch.clamp(torch.abs(left_hip_yaw), min=0.0, max=0.3) - 0.3), dim=0)
        yaw_range_penalty += torch.sum(torch.square(torch.clamp(torch.abs(right_hip_yaw), min=0.0, max=0.3) - 0.3), dim=0)
        
        # 综合惩罚
        total_penalty = roll_pitch_deviation + yaw_symmetry_error + yaw_range_penalty
        return -total_penalty

    def _reward_upper_body_stability(self):
        """上半身稳定性奖励：固定上半身并防止手脚交叉"""
        try:
            # 1. 固定上半身：通过控制髋关节来保持上半身稳定
            # 获取髋关节位置
            left_hip_roll = self.dof_pos[:, 1]   # 左腿hip_roll
            left_hip_pitch = self.dof_pos[:, 2]  # 左腿hip_pitch
            right_hip_roll = self.dof_pos[:, 7]  # 右腿hip_roll
            right_hip_pitch = self.dof_pos[:, 8] # 右腿hip_pitch
            
            # 惩罚髋关节偏离中立位置（固定上半身）
            hip_stability_penalty = torch.square(left_hip_roll) + torch.square(left_hip_pitch) + \
                                   torch.square(right_hip_roll) + torch.square(right_hip_pitch)
            
            # 2. 防止手脚交叉：控制hip_yaw关节的对称性
            left_hip_yaw = self.dof_pos[:, 0]   # 左腿hip_yaw
            right_hip_yaw = self.dof_pos[:, 6]  # 右腿hip_yaw
            
            # 计算hip_yaw的对称性：左右腿hip_yaw应该相反（对称）
            # 理想情况下：left_hip_yaw + right_hip_yaw = 0
            yaw_symmetry_error = torch.square(left_hip_yaw + right_hip_yaw)
            
            # 3. 限制hip_yaw的范围，防止过度旋转
            yaw_range_penalty = torch.square(torch.clamp(torch.abs(left_hip_yaw), min=0.0, max=0.3) - 0.3)
            yaw_range_penalty += torch.square(torch.clamp(torch.abs(right_hip_yaw), min=0.0, max=0.3) - 0.3)
            
            # 4. 鼓励hip_yaw保持小角度（防止过度旋转）
            yaw_magnitude_penalty = torch.square(left_hip_yaw) + torch.square(right_hip_yaw)
            
            # 综合惩罚
            total_penalty = hip_stability_penalty + yaw_symmetry_error + yaw_range_penalty + 0.1 * yaw_magnitude_penalty
            
            return -total_penalty
            
        except Exception as e:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_waist_joint_dev(self):
        """腰部关节偏差奖励: Σ_waist joints |θ_i - θ_i^default|^2"""
        # H1机器人只有12个关节，没有独立的腰部关节
        # 暂时返回零奖励，因为H1是腿部机器人
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_no_fly(self):
        """无飞行奖励: 1{only one feet on ground}"""
        # 计算接触地面的脚数量
        contact_forces = self.contact_forces[:, self.feet_indices, 2]  # z方向力
        feet_on_ground = (contact_forces > 0.1).float()  # 阈值判断
        num_feet_on_ground = torch.sum(feet_on_ground, dim=1)
        
        # 奖励只有一只脚接触地面（行走状态）
        no_fly_reward = (num_feet_on_ground == 1).float()
        return no_fly_reward

    def _reward_feet_lateral_dist(self):
        """脚部横向距离奖励: |y_left foot^B - y_right foot^B - d_min|"""
        # 获取左右脚在身体坐标系中的位置
        left_foot_pos = self.rigid_body_states[:, self.feet_indices[0], 0:3]
        right_foot_pos = self.rigid_body_states[:, self.feet_indices[1], 0:3]
        
        # 计算横向距离
        lateral_distance = torch.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1])
        target_distance = 0.3  # 目标横向距离
        distance_error = torch.abs(lateral_distance - target_distance)
        return -distance_error

    def _reward_feet_slip(self):
        """脚部滑动奖励: Σ_feet |v_i^foot| * ~1_new contact"""
        # 获取脚部速度
        foot_velocities = self.rigid_body_states[:, self.feet_indices, 7:10]
        foot_xy_vel = torch.norm(foot_velocities[:, :, :2], dim=2)
        
        # 检测新接触（简化实现）
        contact_forces = self.contact_forces[:, self.feet_indices, 2]
        new_contact = (contact_forces > 0.1).float()
        
        # 计算滑动惩罚
        slip_penalty = torch.sum(foot_xy_vel * new_contact, dim=1)
        return -slip_penalty

    def _reward_feet_ground_parallel(self):
        """脚部地面平行奖励: 基于脚部方向与垂直方向的偏差"""
        try:
            # 获取脚部方向（四元数）
            foot_orientations = self.rigid_body_states[:, self.feet_indices, 3:7]  # [num_envs, num_feet, 4]
            
            # 简化实现：基于四元数的x,y分量计算倾斜度
            # 当脚部水平时，四元数的x,y分量应该接近0
            # 当脚部倾斜时，x,y分量会增大
            
            x, y = foot_orientations[:, :, 1], foot_orientations[:, :, 2]  # 四元数的x,y分量
            
            # 计算倾斜度：x² + y² 越大说明脚部越倾斜
            tilt_penalty = torch.sum(x**2 + y**2, dim=1)  # [num_envs]
            
            return -tilt_penalty
            
        except Exception as e:
            # 如果计算失败，返回零奖励
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_parallel(self):
        """脚部平行奖励: Var(D)"""
        try:
            # 计算左右脚之间的方向差异
            left_foot_orientation = self.rigid_body_states[:, self.feet_indices[0], 3:7]
            right_foot_orientation = self.rigid_body_states[:, self.feet_indices[1], 3:7]
            
            # 计算方向差异的方差
            orientation_diff = left_foot_orientation - right_foot_orientation
            parallel_variance = torch.var(orientation_diff, dim=1)
            return -parallel_variance
        except:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_contact_momentum(self):
        """接触动量奖励: Σ_feet |v_i^z * F_i^z|"""
        # 获取脚部z方向速度和力
        # rigid_body_states: [pos(0:3), quat(3:7), lin_vel(7:10), ang_vel(10:13)]
        foot_velocities = self.rigid_body_states[:, self.feet_indices, 9]  # z方向线速度
        contact_forces = self.contact_forces[:, self.feet_indices, 2]  # z方向力
        
        # 计算接触动量
        contact_momentum = torch.abs(foot_velocities * contact_forces)
        return -torch.sum(contact_momentum, dim=1)

    def _reward_heading_constraint(self):
        """朝向约束奖励：确保机器人不会朝向后面"""
        try:
            # 获取当前朝向（基于身体yaw）
            current_yaw = self.yaw  # 身体朝向
            
            # 目标朝向：向前（0度）
            target_yaw = torch.zeros_like(current_yaw)
            
            # 计算朝向误差
            yaw_error = torch.abs(current_yaw - target_yaw)
            # 处理角度环绕问题
            yaw_error = torch.min(yaw_error, 2 * torch.pi - yaw_error)
            
            # 如果朝向后面（误差大于π/2），给予严重惩罚
            backward_penalty = torch.where(yaw_error > torch.pi/2, 
                                         torch.exp(yaw_error - torch.pi/2), 
                                         torch.zeros_like(yaw_error))
            
            return -backward_penalty  # 返回负值作为惩罚
            
        except Exception as e:
            return torch.zeros(self.num_envs, device=self.device)
