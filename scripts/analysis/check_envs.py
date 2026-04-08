#!/usr/bin/env python3
"""Check available environments for large-scale experiment."""
import gymnasium as gym

test_envs = [
    # MuJoCo extensions
    'Swimmer-v4', 'Humanoid-v4', 'HumanoidStandup-v4',
    'Reacher-v4', 'InvertedPendulum-v4', 'InvertedDoublePendulum-v4',
    'Pusher-v4',
    # Classic control (always available)
    'Pendulum-v1', 'CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0',
]

print("=== Environment availability check ===")
available = []
for e in test_envs:
    try:
        env = gym.make(e)
        obs, _ = env.reset()
        act = env.action_space
        is_continuous = hasattr(act, 'shape')
        if is_continuous:
            shape_str = str(act.shape)
        else:
            shape_str = f"Discrete({act.n})"
        print(f"  OK  {e:<35} obs={env.observation_space.shape}  act={shape_str}  continuous={is_continuous}")
        if is_continuous:
            available.append(e)
        env.close()
    except Exception as ex:
        print(f"  FAIL {e:<35} -> {ex}")

print(f"\nContinuous-action available: {available}")

