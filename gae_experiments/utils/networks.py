"""
神经网络模块：包含 Actor、Critic 和 Lambda 网络的实现
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """正交初始化"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    """策略网络（Actor）"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, continuous: bool = False):
        super().__init__()
        self.continuous = continuous

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        if continuous:
            self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.logit_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, x: torch.Tensor):
        features = self.net(x)
        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.logit_head(features)
            dist = torch.distributions.Categorical(logits=logits)
        return dist

    def get_action_and_logprob(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(x)
        log_prob = dist.log_prob(actions)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """价值网络（Critic）"""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LambdaNetwork(nn.Module):
    """
    自适应 λ 网络：根据状态动态输出 λ ∈ [0, 1]
    高不确定性 → λ → 0（更信任TD，少展开）
    低不确定性 → λ → 1（更信任MC，多展开）

    改进：
    - 移除末层 Sigmoid，改用带可学习偏置的线性输出后接 Sigmoid，
      允许网络更容易拉开分布（初始偏置 bias_const=1.0 让初始 λ 约 0.73，
      给梯度下降更多空间向两端分化）
    - 网络最后一层使用 std=0.1 而非 0.01，以加大初始输出散布
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 32, init_bias: float = 2.2):
        """
        init_bias=2.2 → sigmoid(2.2)≈0.90，使初始λ接近目标范围 [0.7, 0.99] 的均值
        std=0.05 让各状态的初始输出有一定散布（±0.05 logit，约±0.01 sigmoid）
        """
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            # std=0.05 让初始输出有合理散布，bias_const 控制初始均值
            layer_init(nn.Linear(hidden_dim, 1), std=0.05, bias_const=init_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)

