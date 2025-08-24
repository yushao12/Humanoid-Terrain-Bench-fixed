# 观测结构分析与AMP观测构建策略

## 当前观测结构分析

### 1. 基础观测 (51维)
```
索引 0-2:   base_ang_vel (3维) - 角速度
索引 3-4:   imu_obs (2维) - roll, pitch
索引 5:     delta_yaw (1维) - 目标偏航角差
索引 6:     delta_next_yaw (1维) - 下一个目标偏航角差
索引 7:     commands[0] (1维) - 线速度命令
索引 8:     env_class != 17 (1维) - 环境类别标志
索引 9:     env_class == 17 (1维) - 环境类别标志
索引 10-21: dof_pos (12维) - 关节位置
索引 22-33: dof_vel (12维) - 关节速度
索引 34-45: action_history (12维) - 上一帧动作
索引 46-47: contact_filt (2维) - 接触状态
```

### 2. 历史观测 (510维)
- 10帧 × 51维 = 510维
- 包含过去10帧的完整观测信息

### 3. 特权观测 (38维)
- 显式特权观测: 9维
- 隐式特权观测: 29维

### 4. 扫描观测 (132维)
- 地形高度信息

**总观测维度**: 51 + 132 + 510 + 38 = 816维

## AMP观测构建策略

### 方案1: 利用历史观测构建AMP观测 (推荐)

#### 优势:
1. **不改变网络结构**: 完全基于现有观测
2. **信息丰富**: 历史观测包含完整的运动信息
3. **实现简单**: 只需要从历史观测中提取相关信息

#### 实现方法:
```python
def extract_amp_obs_from_history(obs_history):
    # 从历史观测中提取关节信息
    dof_pos_history = obs_history[:, :, 10:22]  # 12维 × 10帧 = 120维
    dof_vel_history = obs_history[:, :, 22:34]  # 12维 × 10帧 = 120维
    contact_history = obs_history[:, :, 46:48]  # 2维 × 10帧 = 20维
    
    # 计算关节加速度
    dof_acc_history = compute_acceleration(dof_vel_history)
    
    # 组合AMP观测
    amp_obs = concat([dof_pos_history, dof_vel_history, dof_acc_history, contact_history])
    return amp_obs  # 380维
```

### 方案2: 利用特权观测空间

#### 优势:
1. **不影响基础观测**: 保持基础观测的完整性
2. **训练时可用**: 训练时可以使用特权信息

#### 实现方法:
```python
def encode_amp_in_privileged(amp_obs, privileged_obs):
    # 将AMP观测编码到特权观测中
    encoded_privileged = encode_amp_obs(amp_obs, privileged_obs)
    return encoded_privileged
```

## 参考数据适配策略

### SMPL到G1的关节映射

#### SMPL关节结构 (165维):
```
0-2:   根关节旋转 (3维)
3-65:  身体关节pose (21关节 × 3 = 63维)
66-110:左手关节pose (15关节 × 3 = 45维)
111-155:右手关节pose (15关节 × 3 = 45维)
156-158:下巴关节pose (3维)
159-164:眼睛关节pose (6维)
```

#### G1关节结构 (12个DOF):
```
0: left_hip_pitch_joint
1: left_hip_roll_joint
2: left_hip_yaw_joint
3: left_knee_joint
4: left_ankle_pitch_joint
5: left_ankle_roll_joint
6: right_hip_pitch_joint
7: right_hip_roll_joint
8: right_hip_yaw_joint
9: right_knee_joint
10: right_ankle_pitch_joint
11: right_ankle_roll_joint
```

#### 映射关系:
```python
# SMPL下半身关节到G1的映射
smpl_to_g1_mapping = {
    'pelvis': None,  # 根关节，不直接对应
    'L_Hip': [0, 1, 2],  # left_hip_pitch, roll, yaw
    'R_Hip': [6, 7, 8],  # right_hip_pitch, roll, yaw
    'L_Knee': [3],       # left_knee
    'R_Knee': [9],       # right_knee
    'L_Ankle': [4, 5],   # left_ankle_pitch, roll
    'R_Ankle': [10, 11], # right_ankle_pitch, roll
}
```

## 推荐的AMP观测构建流程

### 1. 训练阶段
```python
# 1. 从历史观测中提取AMP观测
amp_obs = extract_amp_obs_from_history(obs_history)

# 2. 从参考数据中提取对应的AMP观测
reference_amp_obs = extract_reference_amp_obs(smpl_data)

# 3. 训练判别器
discriminator_loss = train_discriminator(amp_obs, reference_amp_obs)

# 4. 计算AMP奖励
amp_reward = compute_amp_reward(amp_obs, reference_amp_obs)
```

### 2. 推理阶段
```python
# 1. 从历史观测中提取AMP观测
amp_obs = extract_amp_obs_from_history(obs_history)

# 2. 使用判别器计算奖励
amp_reward = discriminator(amp_obs)
```

## 观测参数总结

### AMP观测维度: 380维
- 关节位置历史: 120维 (12关节 × 10帧)
- 关节速度历史: 120维 (12关节 × 10帧)
- 关节加速度历史: 120维 (12关节 × 10帧)
- 接触状态历史: 20维 (2接触 × 10帧)

### 判别器网络结构
```python
discriminator = nn.Sequential(
    nn.Linear(380, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
```

这个方案的优势是完全基于现有观测结构，不需要修改actor-critic网络，同时能够有效地利用历史观测信息构建AMP观测。 