"""快速冒烟测试：验证所有 Agent 的训练流程"""
import sys
sys.path.insert(0, '.')

from gae_experiments.agents import (
    BasePPO, ConservativeBootstrapPPO, AdaptiveLambdaPPO,
    ConfidenceWeightedPPO, CombinedPPO
)
from gae_experiments.experiment import make_env, set_seed

set_seed(42)
STEPS = 3000
N_STEPS = 512

AGENTS = [
    ("BasePPO", BasePPO),
    ("ConservativeBootstrapPPO", ConservativeBootstrapPPO),
    ("AdaptiveLambdaPPO", AdaptiveLambdaPPO),
    ("ConfidenceWeightedPPO", ConfidenceWeightedPPO),
    ("CombinedPPO", CombinedPPO),
]

for name, AgentClass in AGENTS:
    print(f"\n--- Testing {name} ---")
    env = make_env('CartPole-v1', 42)
    eval_env = make_env('CartPole-v1', 100)
    try:
        agent = AgentClass(env=env, n_steps=N_STEPS, batch_size=64,
                           n_epochs=2, save_dir='/tmp/gae_test')
        logger = agent.train(
            total_timesteps=STEPS,
            eval_env=eval_env,
            eval_freq=1000,
            n_eval_episodes=3,
            verbose=True,
        )
        print(f"  PASSED: eval_rewards={logger.eval_rewards}")
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
    finally:
        env.close()
        eval_env.close()

print("\n\n=== 所有测试完成 ===")

