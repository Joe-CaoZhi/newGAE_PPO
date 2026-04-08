"""从已有实验结果生成可视化图表"""
import sys
sys.path.insert(0, '.')

from gae_experiments.utils.visualizer import (
    load_all_loggers, plot_all_results,
    plot_learning_curves_single, print_summary_table
)

results_dir = "results/CartPole-v1"
env_name = "CartPole-v1"

print(f"从 {results_dir} 加载实验结果...")
loggers = load_all_loggers(results_dir)
print(f"加载到 {len(loggers)} 个实验: {list(loggers.keys())}")

print_summary_table(loggers)

p1 = plot_all_results(loggers, env_name=env_name, save_dir=results_dir)
p2 = plot_learning_curves_single(loggers, env_name=env_name, save_dir=results_dir)

print(f"\n✅ 图表生成完成！")
print(f"  综合对比图: {p1}")
print(f"  学习曲线图: {p2}")

