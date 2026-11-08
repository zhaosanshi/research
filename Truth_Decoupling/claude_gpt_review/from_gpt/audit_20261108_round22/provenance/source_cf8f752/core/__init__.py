"""Truth_Decoupling core library.

Core = model-agnostic, dataset-agnostic machinery:
  config      : JSON config loading with ${ENV} expansion and deep merge
  io_utils    : atomic JSON/JSONL I/O, seeds, experiment naming, checkpoints
  model_io    : model/tokenizer loading, chat prompt building, activation capture
  directions  : direction estimators (mean-diff, logistic probe, rank-r DAS) + random controls
  intervene   : forward-hook steering (add / amplify / interchange / clamp), multi-direction
  layer_score : training-free LA-Score layer selection (dact, dattn, sep, mass)
  readout     : statement-level label readout from logits; abstention detection
  metrics     : J, B, answer rate, dissociation matrix, 3-way decomposition, CIs, tests
  judge       : OpenAI-compatible LLM judge with disk cache (correctness / abstention)
  gate        : knowledge probe + per-input gating of intervention strength

Periphery (configs/, data_adapters/, scripts/) changes when models/datasets change.
Core does not. If a core bug appears, fix it with the smallest possible edit.
"""

__version__ = "0.1.0"
