# Scripts Overview

This folder is organized by responsibility:

- `scripts/models`: model-specific training or inference (CNN, CLAP, PANNs)
- `scripts/tasks`: task-level runners (classification, retrieval, baseline eval)
- `scripts/tools`: utilities (plotting, analysis)

Useful tools:
- `scripts/tools/precompute_features.py`: cache MFCC/log-mel features
- `scripts/tools/compare_librosa.py`: validate DSP vs librosa
- `scripts/models/eval_gemini_zeroshot.py`: Gemini zero-shot baseline
- `scripts/models/eval_ast_transfer.py`: AST transfer baseline
- `scripts/tasks/run_retrieval_ml.py`: retrieval with cnn/panns/ast/clap embeddings

All scripts call APIs under `src/` and keep CLI logic thin.
