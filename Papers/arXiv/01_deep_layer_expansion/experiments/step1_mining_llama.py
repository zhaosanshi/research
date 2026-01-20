"""
Llama-3.3-70B-Instruct-INT8 流形挖掘脚本
使用 NeuralMagic W8A8 量化模型

运行前：
1. pip install compressed-tensors
2. 确保模型已下载到 MODEL_PATH
"""
import torch
import json
import numpy as np

# --- 配置 ---
MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
OUTPUT_FILE = "experiment_data_llama.json"

print("=" * 60)
print("Llama-3.3-70B-Instruct-INT8 流形挖掘")
print("=" * 60)

# --- 加载模型 ---
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f">>> 加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,  # W8A8 模型内部是 INT8，但输出用 FP16
    local_files_only=True
)
print(f">>> 模型加载完成，层数: {model.config.num_hidden_layers}")

# --- 核心数学函数 ---
def compute_intrinsic_dimension(hidden_states):
    """计算内在维度 (Effective Rank)"""
    data = hidden_states.squeeze(0).float().cpu().numpy()
    if data.shape[0] < 2:
        return 0
    try:
        data = data - np.mean(data, axis=0)
        U, S, Vh = np.linalg.svd(data, full_matrices=False)
        S_norm = S / np.sum(S)
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
        return np.exp(entropy)
    except:
        return 0

# --- 实验主循环 ---
def run_experiment():
    with open("topics.json", "r") as f:
        topics = json.load(f)

    results = []

    for i, topic in enumerate(topics):
        print(f"[{i+1}/{len(topics)}] {topic}")

        prompts = {
            "Novice": f"请解释一下 {topic}。",
            "Expert": f"作为该领域的资深专家，请从底层原理和数学推导的角度深度剖析 {topic}。请展示你的思维链。"
        }

        case_data = {"topic": topic, "novice_traj": [], "expert_traj": []}

        for p_type, prompt in prompts.items():
            # Llama 3.3 使用 chat 模板
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            traj = []
            for layer_tensor in outputs.hidden_states:
                dim = compute_intrinsic_dimension(layer_tensor)
                traj.append(float(dim))

            if p_type == "Novice":
                case_data["novice_traj"] = traj
            else:
                case_data["expert_traj"] = traj

        results.append(case_data)

        # 每 10 个保存一次
        if (i + 1) % 10 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f)
            print(f">>> 已保存 {i+1} 条")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print(f">>> 完成！数据保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment()
