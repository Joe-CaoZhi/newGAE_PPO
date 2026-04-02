"""
指标记录器：记录训练过程中的各类指标
"""
import json
import os
from typing import List, Optional

import numpy as np


class MetricLogger:
    """训练指标记录器"""

    def __init__(self, agent_name: str, save_dir: str = "results"):
        self.agent_name = agent_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 主要指标
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_rewards: List[float] = []
        self.eval_steps: List[int] = []

        # 训练辅助指标
        self.value_losses: List[float] = []
        self.policy_losses: List[float] = []
        self.entropy_losses: List[float] = []
        self.approx_kls: List[float] = []
        self.clip_fracs: List[float] = []
        self.explained_variances: List[float] = []

        # 自适应 λ 相关
        self.mean_lambda_values: List[float] = []

        # 步骤计数
        self.total_steps: List[int] = []
        self._current_total_steps = 0

    def log_episode(self, reward: float, length: int):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def log_eval(self, mean_reward: float, step: int):
        self.eval_rewards.append(mean_reward)
        self.eval_steps.append(step)

    def log_update(
        self,
        value_loss: float,
        policy_loss: float,
        entropy_loss: float,
        approx_kl: float,
        clip_frac: float,
        explained_variance: float,
        total_steps: int,
        mean_lambda: Optional[float] = None,
    ):
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)
        self.entropy_losses.append(entropy_loss)
        self.approx_kls.append(approx_kl)
        self.clip_fracs.append(clip_frac)
        self.explained_variances.append(explained_variance)
        self.total_steps.append(total_steps)
        if mean_lambda is not None:
            self.mean_lambda_values.append(mean_lambda)

    def get_recent_reward(self, window: int = 20) -> float:
        if len(self.episode_rewards) == 0:
            return 0.0
        recent = self.episode_rewards[-window:]
        return np.mean(recent)

    def save(self):
        data = {
            "agent_name": self.agent_name,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "eval_steps": self.eval_steps,
            "value_losses": self.value_losses,
            "policy_losses": self.policy_losses,
            "entropy_losses": self.entropy_losses,
            "approx_kls": self.approx_kls,
            "clip_fracs": self.clip_fracs,
            "explained_variances": self.explained_variances,
            "total_steps": self.total_steps,
            "mean_lambda_values": self.mean_lambda_values,
        }
        path = os.path.join(self.save_dir, f"{self.agent_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "MetricLogger":
        with open(path, "r") as f:
            data = json.load(f)
        logger = cls(data["agent_name"])
        logger.episode_rewards = data.get("episode_rewards", [])
        logger.episode_lengths = data.get("episode_lengths", [])
        logger.eval_rewards = data.get("eval_rewards", [])
        logger.eval_steps = data.get("eval_steps", [])
        logger.value_losses = data.get("value_losses", [])
        logger.policy_losses = data.get("policy_losses", [])
        logger.entropy_losses = data.get("entropy_losses", [])
        logger.approx_kls = data.get("approx_kls", [])
        logger.clip_fracs = data.get("clip_fracs", [])
        logger.explained_variances = data.get("explained_variances", [])
        logger.total_steps = data.get("total_steps", [])
        logger.mean_lambda_values = data.get("mean_lambda_values", [])
        return logger

