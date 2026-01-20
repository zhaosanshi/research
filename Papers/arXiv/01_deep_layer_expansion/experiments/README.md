# Deep Layer Expansion Experiments

Experimental code for the paper "Deep Layer Expansion: Expert Prompts Counteract Dimensional Collapse in Large Language Models".

## Files

```
├── step1_mining_llama.py        # Hidden state extraction (Llama-3.3-70B)
├── step1_mining_qwen72b.py      # Hidden state extraction (Qwen2.5-72B)
├── step2_plotting_llama.py      # Visualization (Llama-3.3-70B)
├── step2_plotting_qwen72b.py    # Visualization (Qwen2.5-72B)
├── topics.json                  # 50 technical topics (English)
├── topics_zh.json               # 50 technical topics (Chinese)
├── experiment_data_llama.json   # Raw data (Llama-3.3-70B)
├── experiment_data_qwen72b.json # Raw data (Qwen2.5-72B)
├── Figure_1_Llama70B_Manifold.png/pdf   # Results figure
└── Figure_1_Qwen72B_Manifold.png/pdf    # Results figure
```

## Usage

```bash
# Llama-3.3-70B experiment
python step1_mining_llama.py
python step2_plotting_llama.py

# Qwen2.5-72B experiment
python step1_mining_qwen72b.py
python step2_plotting_qwen72b.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers, numpy, matplotlib, seaborn

## License

MIT
