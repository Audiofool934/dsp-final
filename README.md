# DSP Project: ESC-50 Sound Retrieval and Classification

This project implements FFT/STFT/MFCC from scratch and evaluates:
- Retrieval: fold 5 as queries, folds 1-4 as database, Top-10/Top-20 precision.
- Classification: train on folds 1-4, test on fold 5.
- Retrieval with and without ML embeddings.
- LLM baseline comparison (via external predictions CSV).

## Structure
- `src/dsp`: custom FFT/STFT/MFCC implementations
- `src/datasets`: ESC-50 dataset utilities
- `src/models`: neural network models
- `src/features`: DSP feature cache utilities
- `src/tasks`: task-level pipelines (classification/retrieval)
- `src/retrieval`: retrieval pipelines (MFCC + ML)
- `scripts/models`: model-specific train/infer scripts
- `scripts/tasks`: task-level experiment runners
- `scripts/tools`: plotting and utilities
- `scripts/README.md`: usage notes for scripts layout
- `reports/experiment_plan.md`: full experiment runbook
- `configs`: experiment settings
- `reports`: report template
- `outputs`: results and checkpoints

## Dataset Setup
Download ESC-50 from https://github.com/karolpiczak/ESC-50 and place it at:

```
data/ESC-50-master
```

This directory is ignored by git, with `.gitkeep` as a placeholder.

## Environment
Install dependencies:

```
pip install -r requirements.txt
```

Set `PYTHONPATH` when running scripts:

```
export PYTHONPATH=.
```

## From Scratch Run Order (Brief)
1) Train CNN classifier: `python scripts/models/train_cnn.py --epochs 30`
2) Evaluate retrieval (MFCC): `python scripts/tasks/run_retrieval.py --frame-lengths 512 1024 2048 --hop-lengths 256 512 1024`
3) Retrieval with ML embeddings: `python scripts/tasks/run_retrieval_ml.py --model outputs/models/cnn.pt`
4) CLAP zero-shot baseline: `python scripts/models/eval_clap_zeroshot.py`
5) PANNs transfer baseline: `ESC j`
6) Plot training history: `python scripts/tools/plot_history.py --history outputs/history/train_cnn.csv --output outputs/plots/cnn_history.png`

Full runbook with all experiments: `reports/experiment_plan.md`

## Run All Experiments (Overnight)
This will run all steps in order and write logs/results under `outputs/results/<run_id>`:

```
python scripts/tools/run_all_experiments.py --precompute-workers 1
```

Use `--skip-*` flags to disable heavy steps.
For Gemini prompt variants, use `--gemini-prompt-style guided` to save predictions to a separate file.

To append new results into an existing run:

```
python scripts/tools/continue_run.py --run-id run_YYYYMMDD_HHMMSS
```

## Precompute DSP Features (Cache)
Precompute MFCC and log-mel features and write a manifest:

```
python scripts/tools/precompute_features.py --feature-types mfcc log_mel --workers 4
```

All training and retrieval scripts will read cached features first and compute on cache-miss.

## Retrieval (MFCC)
Run MFCC retrieval with hyperparameter sweeps:

```
python scripts/tasks/run_retrieval.py \
  --frame-lengths 512 1024 2048 \
  --hop-lengths 256 512 1024
```

Results saved to `outputs/retrieval_mfcc.csv`.

## Classification
Train a ResNet-style CNN on log-mel features:

```
python scripts/models/train_cnn.py \
  --frame-length 1024 \
  --hop-length 512 \
  --epochs 30
```

Model saved to `outputs/models/cnn.pt`.
Training history saved to `outputs/history/train_cnn.csv`.

## Classification Hyperparameter Sweep
Compare frame/hop settings on the same ResNet model:

```
python scripts/tasks/run_classification_grid.py --frame-lengths 512 1024 2048 --hop-lengths 256 512 1024
# Use --epochs/--batch-size/--num-workers for a faster grid; defaults are in configs/experiments.yaml
```

## CNN Inference
Run inference on fold 5 and save predictions:

```
python scripts/models/infer_cnn.py --checkpoint outputs/models/cnn.pt
```

## Plot Training History
Plot a history CSV into a PNG:

```
python scripts/tools/plot_history.py --history outputs/history/train_cnn.csv --output outputs/plots/cnn_history.png
```

## Validate DSP vs librosa
Compare custom FFT/STFT/MFCC with librosa on one ESC-50 clip:

```
python scripts/tools/compare_librosa.py --output outputs/validation/librosa_compare.json
```

## Retrieval with ML Embeddings
Use model features for retrieval (cnn/panns/ast/clap):

```
python scripts/tasks/run_retrieval_ml.py \
  --model outputs/models/cnn.pt \
  --model-type cnn \
  --frame-length 1024 \
  --hop-length 512
```

Examples:
```
python scripts/tasks/run_retrieval_ml.py --model-type panns
python scripts/tasks/run_retrieval_ml.py --model-type ast --model-id MIT/ast-finetuned-audioset-10-10-0.4593
python scripts/tasks/run_retrieval_ml.py --model-type clap --model-id laion/clap-htsat-unfused
```

Embedding caching:
- AST/CLAP/PANNs embeddings are cached under `outputs/features/embedding_*` on first run.

## LLM Baseline
Create `outputs/llm_predictions.csv` with columns:

```
filename,predicted_target
1-100032-A-0.wav,0
```

Then evaluate:

```
python scripts/tasks/eval_llm_baseline.py
```

## Gemini Zero-Shot Baseline
Generate predictions with Gemini (requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`):

```
python scripts/models/eval_gemini_zeroshot.py --model gemini-3-flash-preview --sleep 0
```

Guided prompt (single call, summary + label; uses a separate output file when default path is used):

```
python scripts/models/eval_gemini_zeroshot.py --model gemini-3-flash-preview --sleep 0 --prompt-style guided
```

You can increase parallel requests with `--workers` (e.g., `--workers 8`).
Use `--log-every 10` to emit progress lines into logs while running under the experiment runner.

Then evaluate:

```
python scripts/tasks/eval_llm_baseline.py --predictions outputs/llm_predictions.csv
```

For guided runs with the default output path:

```
python scripts/tasks/eval_llm_baseline.py --predictions outputs/llm_predictions_guided.csv
```

## CLAP Zero-Shot Baseline
Run zero-shot classification with CLAP on fold 5:

```
python scripts/models/eval_clap_zeroshot.py --model laion/clap-htsat-unfused
```

Results saved to `outputs/clap_zeroshot.csv`.

## CLAP Transfer Baseline
Extract CLAP audio embeddings and train a linear classifier:

```
python scripts/models/eval_clap_transfer.py
```

Training history saved to `outputs/history/clap_transfer.csv`.
Best linear head saved to `outputs/models/clap_transfer.pt`.

## PANNs Transfer Baseline
Extract PANNs embeddings and train a linear classifier on folds 1-4:

```
python scripts/models/eval_panns_transfer.py
```

Training history saved to `outputs/history/panns_transfer.csv`.
Best linear head saved to `outputs/models/panns_transfer.pt`.

## AST Transfer Baseline
Extract AST embeddings and train a linear classifier on folds 1-4:

```
python scripts/models/eval_ast_transfer.py
```

Training history saved to `outputs/history/ast_transfer.csv`.
Best linear head saved to `outputs/models/ast_transfer.pt`.

## Report
Fill `reports/report_template.md` with experiment settings, curves, and comparisons.
