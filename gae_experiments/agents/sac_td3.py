"""
SAC (Soft Actor-Critic) 和 TD3 (Twin Delayed Deep Deterministic Policy Gradient) 实现
用于与 HCGAE/DCPPO 进行公平的离线策略方法对比

参考文献：
- SAC: Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"
- TD3: Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods"
"""
import time
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 网络模块
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """经验回放缓冲区（用于 SAC 和 TD3）"""

    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[idx]).to(device),
            "actions": torch.FloatTensor(self.actions[idx]).to(device),
            "rewards": torch.FloatTensor(self.rewards[idx]).to(device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(device),
            "dones": torch.FloatTensor(self.dones[idx]).to(device),
        }


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 256, hidden_layers: int = 2) -> nn.Sequential:
    """标准 MLP 网络构建"""
    layers = []
    in_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# SAC 网络组件
# ──────────────────────────────────────────────────────────────────────────────

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACGaussianActor(nn.Module):
    """SAC 高斯策略网络（输出均值和对数标准差）"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, hidden_dim)  # 共享特征提取
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作并计算对数概率（含重参数化技巧 + tanh 压缩）"""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)
        # 计算 log_prob，修正 tanh 的雅可比
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """确定性动作（测试时使用）"""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class SACQNetwork(nn.Module):
    """SAC Q 网络（Twin Q-networks 中的一个）"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# TD3 网络组件
# ──────────────────────────────────────────────────────────────────────────────

class TD3DeterministicActor(nn.Module):
    """TD3 确定性策略网络"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_scale: float = 1.0):
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden_dim)
        self.action_scale = action_scale

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_scale


class TD3QNetwork(nn.Module):
    """TD3 Twin Q 网络（包含两个 Q 函数）"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, 1, hidden_dim)
        self.q2 = mlp(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa)


# ──────────────────────────────────────────────────────────────────────────────
# SAC 智能体
# ──────────────────────────────────────────────────────────────────────────────

class SACAgent:
    """
    Soft Actor-Critic (SAC) 智能体
    采用自动温度调整（alpha 自适应调节）
    """

    NAME = "SAC"

    def __init__(
        self,
        env: gym.Env,
        # 网络
        hidden_dim: int = 256,
        # 算法超参数
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,           # 目标网络软更新系数
        alpha_init: float = 0.2,      # 初始温度系数
        auto_alpha: bool = True,       # 是否自动调整温度
        # 训练
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,  # 预热步数（随机探索）
        update_freq: int = 1,           # 每步更新频率
        n_updates: int = 1,             # 每次更新的梯度步数
        # 评测
        eval_freq: int = 10_240,
        n_eval_episodes: int = 10,
        # 设备和日志
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_freq = update_freq
        self.n_updates = n_updates
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.device = torch.device(device)
        self.save_dir = save_dir

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.action_dim = action_dim

        # 动作范围
        self.action_scale = float(env.action_space.high[0])

        # 网络
        self.actor = SACGaussianActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q1 = SACQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = SACQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = SACQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = SACQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        # 复制参数到目标网络
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # 优化器
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )

        # 自动温度调整
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # 目标熵 = -action_dim（SAC 论文推荐）
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)

        # 评测历史
        self.eval_rewards = []
        self.eval_steps = []

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_deterministic_action(obs_t)
            else:
                action, _ = self.actor.get_action(obs_t)
        action = action.cpu().numpy()[0]
        # 缩放到动作范围
        action = action * self.action_scale
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def update(self) -> Dict[str, float]:
        """从回放缓冲区采样并更新网络"""
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # ── 更新 Q 网络 ──────────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_action(next_obs)
            next_actions = next_actions * self.action_scale
            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1 - dones) * min_q_next

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # ── 更新 Actor ────────────────────────────────────────────────────────
        new_actions, log_pi = self.actor.get_action(obs)
        new_actions_scaled = new_actions * self.action_scale
        q1_pi = self.q1(obs, new_actions_scaled)
        q2_pi = self.q2(obs, new_actions_scaled)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ── 更新温度系数 ─────────────────────────────────────────────────────
        alpha_loss = 0.0
        if self.auto_alpha:
            with torch.no_grad():
                _, log_pi_new = self.actor.get_action(obs)
            alpha_loss = (-self.log_alpha.exp() * (log_pi_new + self.target_entropy)).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        # ── 软更新目标网络 ────────────────────────────────────────────────────
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            "alpha": self.alpha,
        }

    def evaluate(self) -> float:
        """评测当前策略的确定性性能"""
        total_reward = 0.0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            total_reward += ep_reward
        return total_reward / self.n_eval_episodes

    def train(self, total_steps: int, seed: int = 42) -> Dict:
        """主训练循环"""
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs, _ = self.env.reset(seed=seed)
        episode_reward = 0.0
        episode_steps = 0
        n_episodes = 0

        metrics_log = []
        eval_rewards = []
        eval_steps_list = []

        start_time = time.time()
        last_eval_step = 0

        for global_step in range(1, total_steps + 1):
            # 选择动作（预热阶段使用随机动作）
            if global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs)

            # 与环境交互
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

            # 存入回放缓冲区（处理 episode 边界）
            real_done = terminated and not truncated
            self.replay_buffer.add(obs, action, reward, next_obs, float(real_done))
            obs = next_obs

            if done:
                obs, _ = self.env.reset()
                n_episodes += 1
                episode_reward = 0.0
                episode_steps = 0

            # 开始训练更新
            if global_step >= self.learning_starts and global_step % self.update_freq == 0:
                for _ in range(self.n_updates):
                    update_metrics = self.update()

            # 定期评测
            if global_step - last_eval_step >= self.eval_freq:
                eval_reward = self.evaluate()
                eval_rewards.append(eval_reward)
                eval_steps_list.append(global_step)
                last_eval_step = global_step

                elapsed = time.time() - start_time
                print(f"[SAC] Step {global_step:>7d}/{total_steps} | "
                      f"Eval: {eval_reward:>8.1f} | "
                      f"Buffer: {self.replay_buffer.size:>7d} | "
                      f"Elapsed: {elapsed:.0f}s")

        # 最终评测
        final_eval = self.evaluate()

        return {
            "eval_rewards": eval_rewards,
            "eval_steps": eval_steps_list,
            "final_reward": final_eval,
            "last5_mean": float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else final_eval,
        }


# ──────────────────────────────────────────────────────────────────────────────
# TD3 智能体
# ──────────────────────────────────────────────────────────────────────────────

class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) 智能体
    关键特性：
    1. Twin Q-networks（双 Q 网络，减少过估计）
    2. Delayed policy updates（延迟策略更新）
    3. Target policy smoothing（目标策略平滑）
    """

    NAME = "TD3"

    def __init__(
        self,
        env: gym.Env,
        # 网络
        hidden_dim: int = 256,
        # 算法超参数
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,             # 目标网络软更新系数
        policy_noise: float = 0.2,       # 目标策略平滑噪声标准差
        noise_clip: float = 0.5,          # 噪声裁剪范围
        policy_delay: int = 2,            # 策略延迟更新频率
        expl_noise: float = 0.1,          # 探索噪声
        # 训练
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        update_freq: int = 1,
        n_updates: int = 1,
        # 评测
        eval_freq: int = 10_240,
        n_eval_episodes: int = 10,
        # 设备和日志
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_freq = update_freq
        self.n_updates = n_updates
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.device = torch.device(device)
        self.save_dir = save_dir

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.action_dim = action_dim
        self.action_scale = float(env.action_space.high[0])
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # 网络
        self.actor = TD3DeterministicActor(obs_dim, action_dim, hidden_dim, self.action_scale).to(self.device)
        self.actor_target = TD3DeterministicActor(obs_dim, action_dim, hidden_dim, self.action_scale).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TD3QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)

        self._update_step = 0  # 跟踪更新步数（用于延迟策略更新）

    def select_action(self, obs: np.ndarray, noise: bool = True) -> np.ndarray:
        """选择动作，可选择性添加探索噪声"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if noise:
            action += np.random.normal(0, self.expl_noise * self.action_scale, size=action.shape)
        return np.clip(action, self.action_low, self.action_high)

    def update(self) -> Dict[str, float]:
        """从回放缓冲区采样并更新网络"""
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # ── 更新 Critic ──────────────────────────────────────────────────────
        with torch.no_grad():
            # 目标策略平滑：添加裁剪噪声
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_obs) + noise).clamp(
                -self.action_scale, self.action_scale
            )

            # 计算 Twin Q targets
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1_pred, q2_pred = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._update_step += 1

        actor_loss = 0.0
        # ── 延迟策略更新 ─────────────────────────────────────────────────────
        if self._update_step % self.policy_delay == 0:
            actor_loss = -self.critic.q1_value(obs, self.actor(obs)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_loss = actor_loss.item()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss if isinstance(actor_loss, float) else actor_loss.item(),
        }

    def evaluate(self) -> float:
        """评测当前策略的确定性性能"""
        total_reward = 0.0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = self.select_action(obs, noise=False)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            total_reward += ep_reward
        return total_reward / self.n_eval_episodes

    def train(self, total_steps: int, seed: int = 42) -> Dict:
        """主训练循环"""
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs, _ = self.env.reset(seed=seed)
        n_episodes = 0

        eval_rewards = []
        eval_steps_list = []

        start_time = time.time()
        last_eval_step = 0

        for global_step in range(1, total_steps + 1):
            # 预热阶段使用随机动作
            if global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs, noise=True)

            # 与环境交互
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # 存入回放缓冲区
            real_done = terminated and not truncated
            self.replay_buffer.add(obs, action, reward, next_obs, float(real_done))
            obs = next_obs

            if done:
                obs, _ = self.env.reset()
                n_episodes += 1

            # 训练更新
            if global_step >= self.learning_starts and global_step % self.update_freq == 0:
                for _ in range(self.n_updates):
                    self.update()

            # 定期评测
            if global_step - last_eval_step >= self.eval_freq:
                eval_reward = self.evaluate()
                eval_rewards.append(eval_reward)
                eval_steps_list.append(global_step)
                last_eval_step = global_step

                elapsed = time.time() - start_time
                print(f"[TD3] Step {global_step:>7d}/{total_steps} | "
                      f"Eval: {eval_reward:>8.1f} | "
                      f"Buffer: {self.replay_buffer.size:>7d} | "
                      f"Elapsed: {elapsed:.0f}s")

        final_eval = self.evaluate()

        return {
            "eval_rewards": eval_rewards,
            "eval_steps": eval_steps_list,
            "final_reward": final_eval,
            "last5_mean": float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else final_eval,
        }

