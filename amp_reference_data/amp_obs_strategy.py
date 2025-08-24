"""
AMP观测构建策略
基于现有观测结构，不改变actor-critic网络结构
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

class AMPObsStrategy:
    """AMP观测构建策略"""
    
    def __init__(self):
        # G1机器人关节配置
        self.g1_joints = [
            'left_hip_pitch_joint',   # 0
            'left_hip_roll_joint',    # 1  
            'left_hip_yaw_joint',     # 2
            'left_knee_joint',        # 3
            'left_ankle_pitch_joint', # 4
            'left_ankle_roll_joint',  # 5
            'right_hip_pitch_joint',  # 6
            'right_hip_roll_joint',   # 7
            'right_hip_yaw_joint',    # 8
            'right_knee_joint',       # 9
            'right_ankle_pitch_joint',# 10
            'right_ankle_roll_joint', # 11
        ]
        
        # 观测维度分析
        self.obs_dims = {
            'base_obs': 51,
            'scan_obs': 132,
            'history_len': 10,
            'history_obs': 51,
            'privileged_explicit': 9,
            'privileged_latent': 29,
        }
        
        # 历史观测中可用于AMP的部分
        self.history_amp_indices = {
            'dof_pos_start': 15,    # 关节位置在基础观测中的起始索引
            'dof_pos_end': 27,      # 关节位置在基础观测中的结束索引
            'dof_vel_start': 27,    # 关节速度在基础观测中的起始索引  
            'dof_vel_end': 39,      # 关节速度在基础观测中的结束索引
            'contact_start': 49,    # 接触状态在基础观测中的起始索引
            'contact_end': 51,      # 接触状态在基础观测中的结束索引
        }
        
    def extract_amp_obs_from_history(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        从历史观测中提取AMP观测
        
        Args:
            obs_history: 历史观测 (num_envs, history_len, obs_dim)
            
        Returns:
            amp_obs: AMP观测 (num_envs, amp_obs_dim)
        """
        num_envs, history_len, obs_dim = obs_history.shape
        
        # 提取关节位置和速度历史
        dof_pos_history = obs_history[:, :, self.history_amp_indices['dof_pos_start']:self.history_amp_indices['dof_pos_end']]  # (N, 10, 12)
        dof_vel_history = obs_history[:, :, self.history_amp_indices['dof_vel_start']:self.history_amp_indices['dof_vel_end']]  # (N, 10, 12)
        contact_history = obs_history[:, :, self.history_amp_indices['contact_start']:self.history_amp_indices['contact_end']]  # (N, 10, 2)
        
        # 计算关节加速度 (使用有限差分)
        dof_acc_history = torch.zeros_like(dof_vel_history)
        dof_acc_history[:, 1:] = (dof_vel_history[:, 1:] - dof_vel_history[:, :-1]) / 0.033  # 假设30fps
        
        # 构建AMP观测
        amp_obs = torch.cat([
            dof_pos_history.view(num_envs, -1),    # 120维 (10帧 × 12关节)
            dof_vel_history.view(num_envs, -1),    # 120维 (10帧 × 12关节)  
            dof_acc_history.view(num_envs, -1),    # 120维 (10帧 × 12关节)
            contact_history.view(num_envs, -1),    # 20维 (10帧 × 2接触)
        ], dim=-1)
        
        return amp_obs  # 总共380维
    
    def convert_smpl_to_g1_obs(self, smpl_data: Dict) -> torch.Tensor:
        """
        将SMPL参考数据转换为g1观测格式
        
        Args:
            smpl_data: SMPL数据字典
            
        Returns:
            g1_obs: g1格式的观测
        """
        poses = smpl_data['poses']  # (N, 165)
        trans = smpl_data['trans']  # (N, 3)
        
        # 提取下半身关节pose (对应g1的12个DOF)
        # SMPL body pose: 21个关节 × 3 = 63维
        body_pose = poses[:, 3:66]
        
        # 映射到g1的下半身关节
        # 这里需要根据具体的关节映射关系
        # 暂时使用前12个关节作为下半身
        g1_pose = body_pose[:, :36]  # 12个关节 × 3 = 36维
        
        # 转换为g1观测格式
        g1_obs = torch.cat([
            trans,      # 3维 (根关节位置)
            g1_pose,    # 36维 (关节pose)
            torch.zeros(poses.shape[0], 12),  # 12维 (关节速度占位符)
        ], dim=-1)
        
        return g1_obs
    
    def build_amp_reference_dataset(self, motion_files: List[str]) -> torch.Tensor:
        """
        构建AMP参考数据集
        
        Args:
            motion_files: 运动文件列表
            
        Returns:
            amp_dataset: AMP参考数据集
        """
        amp_data = []
        
        for motion_file in motion_files:
            # 加载SMPL数据
            smpl_data = np.load(motion_file, allow_pickle=True)
            
            # 转换为g1格式
            g1_obs = self.convert_smpl_to_g1_obs(smpl_data)
            
            # 构建历史观测格式
            num_frames = g1_obs.shape[0]
            history_obs = []
            
            for i in range(num_frames):
                # 构建单帧观测 (51维)
                frame_obs = torch.zeros(51)
                
                # 填充关节位置 (12维)
                frame_obs[15:27] = g1_obs[i, 3:15]  # 关节pose的前12维
                
                # 填充关节速度 (12维) - 使用有限差分计算
                if i > 0:
                    frame_obs[27:39] = (g1_obs[i, 3:15] - g1_obs[i-1, 3:15]) / 0.033
                
                # 填充接触状态 (2维) - 暂时设为0
                frame_obs[49:51] = 0
                
                history_obs.append(frame_obs)
            
            # 构建历史观测
            history_tensor = torch.stack(history_obs)
            amp_data.append(history_tensor)
        
        return torch.cat(amp_data, dim=0)
    
    def compute_amp_reward(self, current_obs: torch.Tensor, reference_obs: torch.Tensor) -> torch.Tensor:
        """
        计算AMP奖励
        
        Args:
            current_obs: 当前观测
            reference_obs: 参考观测
            
        Returns:
            amp_reward: AMP奖励
        """
        # 提取关节信息
        current_dof_pos = current_obs[:, self.history_amp_indices['dof_pos_start']:self.history_amp_indices['dof_pos_end']]
        reference_dof_pos = reference_obs[:, self.history_amp_indices['dof_pos_start']:self.history_amp_indices['dof_pos_end']]
        
        # 计算关节位置误差
        dof_error = torch.norm(current_dof_pos - reference_dof_pos, dim=-1)
        
        # 计算AMP奖励 (使用负误差作为奖励)
        amp_reward = -dof_error
        
        return amp_reward

# 使用示例
if __name__ == "__main__":
    strategy = AMPObsStrategy()
    
    # 模拟历史观测数据
    num_envs = 4
    history_len = 10
    obs_dim = 51
    
    obs_history = torch.randn(num_envs, history_len, obs_dim)
    
    # 提取AMP观测
    amp_obs = strategy.extract_amp_obs_from_history(obs_history)
    print(f"AMP观测维度: {amp_obs.shape}")
    
    # 构建参考数据集
    motion_files = [
        'amp_reference_data/ACCAD/B1_-_stand_to_walk_stageii.npz',
        'amp_reference_data/ACCAD/B2_-_walk_to_stand_stageii.npz',
    ]
    
    # amp_dataset = strategy.build_amp_reference_dataset(motion_files)
    # print(f"AMP参考数据集维度: {amp_dataset.shape}") 