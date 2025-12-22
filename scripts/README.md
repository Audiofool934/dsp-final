# Scripts Overview

This folder is organized by responsibility:

- `scripts/models`: model-specific training or inference (CNN, CLAP, PANNs)
- `scripts/tasks`: task-level runners (classification, retrieval, baseline eval)
- `scripts/tools`: utilities (plotting, analysis)

Useful tools:
- `scripts/tools/precompute_features.py`: cache MFCC/log-mel features
- `scripts/tools/compare_librosa.py`: validate DSP vs librosa
- `scripts/models/eval_gemini_zeroshot.py`: Gemini zero-shot baseline (`--prompt-style guided` for summary+label)
- `scripts/models/eval_ast_transfer.py`: AST transfer baseline
- `scripts/models/eval_clap_transfer.py`: CLAP transfer baseline
- `scripts/models/train_cnn.py`: ResNet-style classifier (from scratch)
- `scripts/models/infer_transfer.py`: inference for AST/PANNs/CLAP linear probes
- `scripts/tasks/run_retrieval_ml.py`: retrieval with cnn/panns/ast/clap embeddings
- `scripts/tools/run_all_experiments.py`: end-to-end experiment runner
- `scripts/tools/continue_run.py`: resume runs in the same output directory
- `scripts/tools/normalize_llm_predictions.py`: normalize Gemini outputs and recompute accuracy
- `scripts/tools/export_prediction_errors.py`: export per-model error cases from prediction CSVs
- `scripts/tools/analyze_error_sets.py`: copy error audio and summarize error patterns
- `scripts/tools/compare_llm_prompts.py`: compare LLM prompt variants and write summary

All scripts call APIs under `src/` and keep CLI logic thin.
