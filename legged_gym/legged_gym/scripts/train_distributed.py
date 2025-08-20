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

import numpy as np
import os
import signal
import sys
from datetime import datetime

# 先导入Isaac Gym相关模块
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

# 然后导入PyTorch
import torch
import torch.distributed as dist

# 禁用wandb
# import wandb

def signal_handler(signum, frame):
    """处理中断信号，确保优雅退出"""
    print(f"\n收到信号 {signum}，正在优雅退出...")
    
    # 清理分布式训练
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("分布式进程组已清理")
        except Exception as e:
            print(f"清理分布式进程组时出错: {e}")
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("CUDA缓存已清理")
        except Exception as e:
            print(f"清理CUDA缓存时出错: {e}")
    
    print("退出完成")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill命令

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 检查CUDA设备是否可用
        if not torch.cuda.is_available():
            print(f"CUDA not available for rank {rank}")
            return rank, world_size, 0
        
        # 检查设备数量
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            print(f"Local rank {local_rank} >= device count {device_count}")
            return rank, world_size, 0
        
        try:
            # 初始化进程组
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            
            # 确保每个进程使用正确的GPU
            torch.cuda.set_device(local_rank)
            
            print(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}, device=cuda:{local_rank}")
            return rank, world_size, local_rank
        except Exception as e:
            print(f"Failed to setup distributed training for rank {rank}: {e}")
            return rank, world_size, 0
    else:
        print("Not using distributed training")
        return 0, 1, 0

def check_gpu_memory(local_rank, num_envs):
    """检查GPU内存是否足够"""
    try:
        torch.cuda.set_device(local_rank)
        total_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(local_rank) / 1024**3  # GB
        
        # 估算每个环境需要的内存（经验值）
        estimated_memory_per_env = 0.008  # GB per env (8卡时每个环境更少)
        required_memory = num_envs * estimated_memory_per_env
        
        print(f"GPU {local_rank}: Total={total_memory:.1f}GB, Allocated={allocated_memory:.1f}GB, Required={required_memory:.1f}GB")
        
        if required_memory > total_memory * 0.8:  # 使用80%作为安全阈值
            print(f"Warning: GPU {local_rank} may not have enough memory for {num_envs} environments")
            return False
        return True
    except Exception as e:
        print(f"Error checking GPU {local_rank} memory: {e}")
        return False

def train_distributed(args):
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    # 只在主进程上创建日志目录
    if rank == 0:
        log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid
        try:
            os.makedirs(log_pth)
        except:
            pass
    else:
        # 非主进程也设置日志路径，但使用临时目录
        log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid + f"_rank{rank}"
        try:
            os.makedirs(log_pth)
        except:
            pass
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"
    
    # 强制禁用wandb
    mode = "disabled"
    args.no_wandb = True

    # 调整环境数量以适应分布式训练
    if dist.is_initialized():
        # 将总环境数分配给所有GPU
        total_envs = args.num_envs
        args.num_envs = total_envs // world_size
        
        # 检查GPU内存
        if not check_gpu_memory(local_rank, args.num_envs):
            print(f"GPU {local_rank}: Reducing environments from {args.num_envs} to {args.num_envs // 2}")
            args.num_envs = args.num_envs // 2
        
        print(f"GPU {rank}: using {args.num_envs} environments out of {total_envs} total")
        
        # 设置每个进程使用对应的GPU - 确保一致性
        device_str = f"cuda:{local_rank}"
        args.device = device_str
        args.rl_device = device_str
        args.sim_device = device_str
        print(f"Process {rank} using device: {args.device}")

    try:
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
        
        # 设置分布式训练 - 延迟包装
        if dist.is_initialized():
            # 等待所有进程完成环境创建
            dist.barrier()
            
            # 只包装需要梯度同步的网络层，而不是整个网络
            if hasattr(ppo_runner.alg, 'actor_critic'):
                # 包装actor的backbone网络（主要计算部分）
                if hasattr(ppo_runner.alg.actor_critic.actor, 'actor_backbone'):
                    ppo_runner.alg.actor_critic.actor.actor_backbone = torch.nn.parallel.DistributedDataParallel(
                        ppo_runner.alg.actor_critic.actor.actor_backbone, 
                        device_ids=[local_rank],
                        output_device=local_rank
                    )
                
                # 包装critic网络
                if hasattr(ppo_runner.alg.actor_critic, 'critic'):
                    ppo_runner.alg.actor_critic.critic = torch.nn.parallel.DistributedDataParallel(
                        ppo_runner.alg.actor_critic.critic, 
                        device_ids=[local_rank],
                        output_device=local_rank
                    )
        
        # 重写save方法，在保存前移除DDP包装
        original_save = ppo_runner.save
        def save_without_ddp(path, infos=None):
            # 临时移除DDP包装
            original_actor_backbone = None
            original_critic = None
            
            if hasattr(ppo_runner.alg.actor_critic.actor, 'actor_backbone') and hasattr(ppo_runner.alg.actor_critic.actor.actor_backbone, 'module'):
                original_actor_backbone = ppo_runner.alg.actor_critic.actor.actor_backbone
                ppo_runner.alg.actor_critic.actor.actor_backbone = ppo_runner.alg.actor_critic.actor.actor_backbone.module
            
            if hasattr(ppo_runner.alg.actor_critic, 'critic') and hasattr(ppo_runner.alg.actor_critic.critic, 'module'):
                original_critic = ppo_runner.alg.actor_critic.critic
                ppo_runner.alg.actor_critic.critic = ppo_runner.alg.actor_critic.critic.module
            
            # 保存模型
            original_save(path, infos)
            
            # 恢复DDP包装
            if original_actor_backbone is not None:
                ppo_runner.alg.actor_critic.actor.actor_backbone = original_actor_backbone
            if original_critic is not None:
                ppo_runner.alg.actor_critic.critic = original_critic
        
        ppo_runner.save = save_without_ddp
        
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
        
    except Exception as e:
        print(f"Error in training process {rank}: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
    finally:
        # 确保清理资源
        if dist.is_initialized():
            try:
                # 同步所有进程
                dist.barrier()
                
                # 只在主进程保存最终模型
                if rank == 0 and log_pth is not None:
                    print(f"主进程保存最终模型到: {log_pth}")
                    # 保存最终模型（移除DDP包装后的原始模型）
                    final_model_path = os.path.join(log_pth, 'model_final.pt')
                    ppo_runner.save(final_model_path)
                    print(f"最终模型已保存到: {final_model_path}")
                
                # 所有进程都保存最终模型到各自的目录
                if log_pth is not None:
                    print(f"Process {rank} 保存最终模型到: {log_pth}")
                    # 保存最终模型（移除DDP包装后的原始模型）
                    final_model_path = os.path.join(log_pth, 'model_final.pt')
                    ppo_runner.save(final_model_path)
                    print(f"Process {rank} 最终模型已保存到: {final_model_path}")
                
                dist.destroy_process_group()
                print(f"Process {rank}: 分布式进程组已清理")
            except:
                pass
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print(f"Process {rank}: CUDA缓存已清理")
            except:
                pass

if __name__ == '__main__':
    args = get_args()
    train_distributed(args) 