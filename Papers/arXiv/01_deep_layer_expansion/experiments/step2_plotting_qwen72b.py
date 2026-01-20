"""
Qwen2.5-72B 流形可视化
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5
})

INPUT_FILE = "experiment_data_qwen72b.json"
OUTPUT_PNG = "Figure_1_Qwen72B_Manifold.png"
OUTPUT_PDF = "Figure_1_Qwen72B_Manifold.pdf"

def plot_manifold():
    print(">>> 绘制 Qwen2.5-72B 流形轨迹图...")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print(f">>> 读取 {len(data)} 条数据")

    novice_matrix = np.array([item['novice_traj'] for item in data])
    expert_matrix = np.array([item['expert_traj'] for item in data])

    layers = np.arange(novice_matrix.shape[1])
    num_layers = len(layers)

    nov_mean = np.mean(novice_matrix, axis=0)
    nov_std = np.std(novice_matrix, axis=0) / np.sqrt(len(data))
    exp_mean = np.mean(expert_matrix, axis=0)
    exp_std = np.std(expert_matrix, axis=0) / np.sqrt(len(data))

    diff_mean = exp_mean - nov_mean

    # 跳过前 10 层（Embedding 层伪影），找真正有意义的最大差异
    valid_start = 10
    max_diff_layer = valid_start + np.argmax(diff_mean[valid_start:])
    max_diff_value = diff_mean[max_diff_layer]

    # 找深层膨胀区的代表点（Layer 60-75）
    deep_layer = 70
    deep_diff = diff_mean[deep_layer]
    deep_ratio = (exp_mean[deep_layer] / nov_mean[deep_layer] - 1) * 100

    print(f">>> 层数: {num_layers}, 最大差异层: {max_diff_layer}, 差值: {max_diff_value:.2f}")
    print(f">>> Layer {deep_layer} 差异: {deep_diff:.2f} dims, 提升 {deep_ratio:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(layers, nov_mean, label="Standard Prompt (Baseline)",
            color="#3498db", linestyle="--", marker='o', markersize=3)
    ax.fill_between(layers, nov_mean - nov_std, nov_mean + nov_std, color="#3498db", alpha=0.2)

    ax.plot(layers, exp_mean, label="Expert Prompt (Manifold Teleportation)",
            color="#e74c3c", marker='s', markersize=3)
    ax.fill_between(layers, exp_mean - exp_std, exp_mean + exp_std, color="#e74c3c", alpha=0.2)

    # 标注深层膨胀区
    ax.annotate(f'Deep Layer Expansion\n(+{deep_ratio:.0f}% dimension)',
                xy=(deep_layer, exp_mean[deep_layer]),
                xytext=(deep_layer - 15, exp_mean[deep_layer] + 12),
                arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=1.5, edgecolor='none'),
                ha='center', fontsize=11, color='#c0392b')

    # 标注分叉起点
    diverge_layer = 30
    ax.annotate('Trajectory Divergence',
                xy=(diverge_layer, exp_mean[diverge_layer]),
                xytext=(diverge_layer + 10, exp_mean[diverge_layer] + 8),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, edgecolor='none'),
                ha='center', fontsize=10, color='gray')

    ax.set_xlabel("Network Depth (Layers)")
    ax.set_ylabel("Effective Intrinsic Dimension")
    ax.set_title("Manifold Geometry of Qwen2.5-72B-Instruct Reasoning")
    ax.legend(loc="upper right")

    ax.text(0.02, 0.98, f"Model: Qwen2.5-72B-Instruct-AWQ\nLayers: {num_layers}\nTopics: {len(data)}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.savefig(OUTPUT_PDF)
    print(f">>> 生成: {OUTPUT_PNG}")

if __name__ == "__main__":
    plot_manifold()
