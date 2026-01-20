# DeepSeek Manifold Geometry Experiment

Probing the **manifold geometry trajectories** of DeepSeek reasoning models under different prompts to verify whether "Expert Prompts" trigger higher-dimensional thought space expansion.

## Core Hypothesis

The hidden layer representations of LLMs exhibit dynamic changes in **Intrinsic Dimension (ID)**:
- **Standard Prompt**: Shallow divergence → Deep convergence (conventional pattern)
- **Expert Prompt + CoT**: **Geometric expansion** in middle layers (hourglass effect), activating more complex reasoning paths

## Experimental Design

| Condition | Prompt Template |
|-----------|-----------------|
| Novice (Baseline) | `Please explain {topic}.` |
| Expert (Treatment) | `As a senior expert in this field, please analyze {topic} in depth from the perspective of underlying principles and mathematical derivations. Show your chain of thought.` |

Testing 50 technical questions in computer science, extracting the Effective Rank (intrinsic dimension) at each layer.

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+ (MPS/CUDA support)
- 8GB+ VRAM (Mac) or 24GB+ VRAM (NVIDIA)

### Installation

```bash
# Create virtual environment
conda create -n deepseek_env python=3.10
conda activate deepseek_env

# Install dependencies
pip install torch transformers numpy matplotlib seaborn

# (Optional) Install ModelScope for faster downloads in China
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Run Experiment

```bash
# Step 1: Extract hidden layer data
python step1_mining.py

# Step 2: Generate visualizations
python step2_plotting.py
```

## File Structure

```
.
├── step1_mining.py          # Data mining script (extract intrinsic dimensions)
├── step2_plotting.py        # Visualization script (generate paper figures)
├── topics.json              # 50 test questions
├── topics_5.json            # 5 test questions (for quick validation)
├── experiment_data.json     # Raw experimental data
├── Figure_1_DeepSeek_Manifold.png  # Output figure (PNG)
└── Figure_1_DeepSeek_Manifold.pdf  # Output figure (PDF)
```

## Core Algorithm

**Intrinsic Dimension Calculation (Effective Rank)**:

```python
def compute_intrinsic_dimension(hidden_states):
    # SVD decomposition
    U, S, Vh = np.linalg.svd(data, full_matrices=False)
    # Normalize singular values
    S_norm = S / np.sum(S)
    # Shannon entropy
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
    # Effective dimension = exp(Entropy)
    return np.exp(entropy)
```

This metric reflects the **effective degrees of freedom** of hidden layer representations:
- High ID → Divergent thinking, exploring multiple reasoning paths
- Low ID → Convergent thinking, focusing on a single answer

## Hardware Compatibility

| Device | Model | VRAM Required |
|--------|-------|---------------|
| Mac M1/M2/M3/M4 | DeepSeek-R1-Distill-Qwen-1.5B | ~4GB |
| NVIDIA GPU | DeepSeek-R1-Distill-Llama-70B | ~140GB (multi-GPU) |

The script automatically detects hardware and selects the appropriate model.

## Model Download

The script supports two download sources:

1. **ModelScope (Alibaba Cloud)** - Preferred for users in China, faster
2. **HuggingFace Mirror** - Backup option

If download fails, you can manually set the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Expected Results

The experiment should observe:

1. **Expert Prompt's middle layer dimensions significantly higher than Novice** (geometric expansion)
2. **Two curves converge at first and last layers** (input/output layer dimensions are constrained)
3. **Expert curve shows a "bulge" in middle layers** (CoT activation signature)

## Citation

If this experiment is helpful for your research, please cite:

```bibtex
@misc{deepseek_manifold_2025,
  title={Manifold Geometry of DeepSeek Reasoning},
  author={Zhao Lei},
  year={2025},
  howpublished={\url{https://github.com/...}}
}
```

## License

MIT
