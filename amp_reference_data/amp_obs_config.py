"""
AMP (Adversarial Motion Priors) 观测配置
用于g1机器人的下半身运动控制
"""

import numpy as np
import torch

class AMPObsConfig:
    """AMP观测配置类"""
    
    def __init__(self):
        # G1机器人下半身关节配置 (12个DOF)
        self.g1_lower_body_joints = [
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
        
        # SMPL下半身关节映射 (对应g1的12个DOF)
        self.smpl_lower_body_indices = [
            0,   # pelvis
            1,   # L_hip  
            2,   # R_hip
            4,   # L_knee
            5,   # R_knee
            7,   # L_ankle
            8,   # R_ankle
            10,  # L_foot
            11,  # R_foot
        ]
        
        # 观测维度配置
        self.obs_dims = {
            # 基础观测 (51维，来自当前环境)
            'base_obs': 51,
            
            # AMP参考运动观测
            'amp_reference_obs': {
                'root_pos': 3,        # 根关节位置
                'root_rot': 4,        # 根关节旋转 (四元数)
                'lower_body_pos': 12, # 下半身关节位置 (12个DOF)
                'lower_body_vel': 12, # 下半身关节速度 (12个DOF)
                'lower_body_acc': 12, # 下半身关节加速度 (12个DOF)
                'contact_state': 2,   # 接触状态 (左右脚)
            },
            
            # 历史观测
            'history_len': 10,        # 历史长度
            'history_obs': 51,        # 历史观测维度
            
            # 特权观测
            'privileged_obs': {
                'explicit': 9,        # 显式特权观测
                'latent': 29,         # 隐式特权观测
            }
        }
        
        # 计算总观测维度
        self.total_obs_dim = (
            self.obs_dims['base_obs'] +  # 51
            sum(self.obs_dims['amp_reference_obs'].values()) +  # 45
            self.obs_dims['history_len'] * self.obs_dims['history_obs'] +  # 510
            sum(self.obs_dims['privileged_obs'].values())  # 38
        )
        
        # 观测缩放因子
        self.obs_scales = {
            'root_pos': 1.0,
            'root_rot': 1.0,
            'joint_pos': 1.0,
            'joint_vel': 0.05,
            'joint_acc': 0.1,
            'lin_vel': 2.0,
            'ang_vel': 0.25,
        }
        
        # AMP参考运动配置
        self.amp_config = {
            'motion_file_dir': 'amp_reference_data/ACCAD',
            'motion_files': [
                'B1_-_stand_to_walk_stageii.npz',
                'B2_-_walk_to_stand_stageii.npz',
                'B3_-_walk1_stageii.npz',
                'B4_-_stand_to_walk_back_stageii.npz',
                'B5_-_walk_backwards_stageii.npz',
                'B6_-_walk_backwards_to_stand_stageii.npz',
                'B7_-_walk_backwards_turn_forwards_stageii.npz',
                'B9_-_walk_turn_left_(90)_stageii.npz',
                'B10_-_walk_turn_left_(45)_stageii.npz',
                'B11_-_walk_turn_left_(135)_stageii.npz',
                'B12_-_walk_turn_right_(90)_stageii.npz',
                'B13_-_walk_turn_right_(45)_stageii.npz',
                'B14_-_walk_turn_right_(135)_stageii.npz',
                'B15_-_walk_turn_around_(same_direction)_stageii.npz',
                'B16_-_walk_turn_change_direction_stageii.npz',
                'B17_-_walk_to_hop_to_walk1_stageii.npz',
                'B18_-_walk_to_leap_to_walk_stageii.npz',
                'B19_-_walk_to_pick_up_box_stageii.npz',
                'B20_-_walk_with_box_stageii.npz',
                'B21_-_put_down_box_to_walk_stageii.npz',
                'B22_-_side_step_left_stageii.npz',
                'B23_-_side_step_right_stageii.npz',
                'B24_-_walk_to_crouch_stageii.npz',
                'B25_-_crouch_to_walk1_stageii.npz',
                'B27_-skip_to_walk1_stageii.npz',
                'General_A1_-_Stand_stageii.npz',
            ],
            'motion_fps': 30,
            'motion_scale': 1.0,
            'phase_encoding': True,  # 是否使用相位编码
            'phase_dim': 2,          # 相位编码维度
        }
        
        # 判别器网络配置
        self.discriminator_config = {
            'hidden_dims': [512, 256, 128],
            'input_dim': 45,  # AMP参考观测维度
            'output_dim': 1,  # 判别器输出维度
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
        }
        
    def get_amp_reference_obs_dim(self):
        """获取AMP参考观测维度"""
        return sum(self.obs_dims['amp_reference_obs'].values())
    
    def get_total_obs_dim(self):
        """获取总观测维度"""
        return self.total_obs_dim
    
    def extract_lower_body_from_smpl(self, smpl_poses, smpl_trans):
        """
        从SMPL全身体pose中提取下半身关节信息
        
        Args:
            smpl_poses: SMPL pose参数 (N, 165)
            smpl_trans: SMPL根关节位置 (N, 3)
            
        Returns:
            lower_body_obs: 下半身观测 (N, 45)
        """
        # 提取根关节信息
        root_orient = smpl_poses[:, :3]  # 根关节旋转 (轴角)
        root_pos = smpl_trans  # 根关节位置
        
        # 提取下半身关节pose (对应SMPL的body pose部分)
        body_pose = smpl_poses[:, 3:66]  # 21个关节 * 3 = 63维
        
        # 映射到g1的下半身关节
        # 这里需要根据具体的关节映射关系来提取
        # 暂时使用前12个关节作为下半身
        lower_body_pose = body_pose[:, :36]  # 12个关节 * 3 = 36维
        
        # 转换为四元数
        root_quat = self.axis_angle_to_quaternion(root_orient)
        
        # 组合观测
        lower_body_obs = torch.cat([
            root_pos,           # 3维
            root_quat,          # 4维  
            lower_body_pose,    # 36维
            torch.zeros(smpl_poses.shape[0], 2),  # 接触状态占位符
        ], dim=-1)
        
        return lower_body_obs
    
    def axis_angle_to_quaternion(self, axis_angle):
        """轴角转四元数"""
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)
        axis = axis_angle / (angle + 1e-8)
        
        w = torch.cos(angle / 2)
        xyz = torch.sin(angle / 2) * axis
        
        return torch.cat([w, xyz], dim=-1)
    
    def compute_phase_encoding(self, time_step, motion_length):
        """计算相位编码"""
        phase = 2 * np.pi * time_step / motion_length
        return np.array([np.cos(phase), np.sin(phase)])

# 创建全局配置实例
amp_obs_config = AMPObsConfig()

if __name__ == "__main__":
    print("AMP观测配置:")
    print(f"总观测维度: {amp_obs_config.get_total_obs_dim()}")
    print(f"AMP参考观测维度: {amp_obs_config.get_amp_reference_obs_dim()}")
    print(f"G1下半身关节数: {len(amp_obs_config.g1_lower_body_joints)}")
    print(f"参考运动文件数: {len(amp_obs_config.amp_config['motion_files'])}") 