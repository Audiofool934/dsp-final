# Experiment Runbook (Logical Order)

This document lists all experiments (run or not) in a clear, repeatable order.

## 0) Environment
```
pip install -r requirements.txt
export PYTHONPATH=.
```

## 1) DSP validation (custom vs librosa)
Purpose: sanity-check FFT/STFT/MFCC alignment.
```
python scripts/tools/compare_librosa.py --output outputs/validation/librosa_compare.json
```
Expected output: `outputs/validation/librosa_compare.json`

## 2) Precompute features (optional but recommended)
### 2.1 MFCC for Task 1 retrieval (all frame/hop combos)
```
for fl in 512 1024 2048; do
  for hl in 256 512 1024; do
    python scripts/tools/precompute_features.py \
      --feature-types mfcc \
      --frame-length $fl --hop-length $hl \
      --workers 1
  done
done
```

### 2.2 Log-mel for CNN training (default params)
```
python scripts/tools/precompute_features.py --feature-types log_mel --workers 1
```

Cache location: `outputs/features/*`

## 3) Task 1: Retrieval (MFCC)
Purpose: evaluate Top-10/Top-20 precision with different frame/hop settings.
```
python scripts/tasks/run_retrieval.py --frame-lengths 512 1024 2048 --hop-lengths 256 512 1024
```
Expected output: `outputs/retrieval_mfcc.csv`

## 4) Task 2: Classification (CNN, custom log-mel)
```
python scripts/models/train_cnn.py --epochs 30
```
Expected outputs:
- `outputs/models/cnn.pt`
- `outputs/history/train_cnn.csv`

## 4.1) Classification Hyperparameter Sweep (frame/hop)
```
python scripts/tasks/run_classification_grid.py --frame-lengths 512 1024 2048 --hop-lengths 256 512 1024
Optional: use --epochs/--batch-size/--num-workers for a faster sweep (defaults in configs/experiments.yaml).
```
Expected output: `outputs/classification_grid.csv` (direct run) or `outputs/results/<run>/history/classification_grid.csv` (via run_resnet_update)

## 5) Retrieval with ML embeddings (CNN)
```
python scripts/tasks/run_retrieval_ml.py --model outputs/models/cnn.pt --model-type cnn
```

## 6) Transfer baselines (classification)
### 6.1 PANNs transfer (linear probe)
```
python scripts/models/eval_panns_transfer.py
```
Expected: `outputs/history/panns_transfer.csv`

### 6.2 AST transfer (linear probe)
```
python scripts/models/eval_ast_transfer.py
```
Expected: `outputs/history/ast_transfer.csv`

### 6.3 CLAP transfer (linear probe)
```
python scripts/models/eval_clap_transfer.py
```
Expected: `outputs/history/clap_transfer.csv`

## 7) Retrieval with pretrained embeddings (PANNs / AST / CLAP)
```
python scripts/tasks/run_retrieval_ml.py --model-type panns
python scripts/tasks/run_retrieval_ml.py --model-type ast --model-id MIT/ast-finetuned-audioset-10-10-0.4593
python scripts/tasks/run_retrieval_ml.py --model-type clap --model-id laion/clap-htsat-unfused
```
Embeddings are cached under:
- `outputs/features/embedding_panns/*`
- `outputs/features/embedding_ast/*`
- `outputs/features/embedding_clap/*`

## 8) Zero-shot baselines (LLM / CLAP)
### 8.1 CLAP zero-shot classification
```
python scripts/models/eval_clap_zeroshot.py --model laion/clap-htsat-unfused --batch-size 8
```
Expected output: `outputs/clap_zeroshot.csv`

### 8.2 Gemini zero-shot classification (optional)
Set API key:
```
export GOOGLE_API_KEY=...
```
Run:
```
python scripts/models/eval_gemini_zeroshot.py --model gemini-3-flash-preview --sleep 0.5
```
Evaluate:
```
python scripts/tasks/eval_llm_baseline.py --predictions outputs/llm_predictions.csv
```

## 9) Plot training history
```
python scripts/tools/plot_history.py --history outputs/history/train_cnn.csv --output outputs/plots/cnn_history.png
```

## 10) Report assembly checklist
- DSP validation metrics (librosa comparison)
- Retrieval table (Top-10/Top-20)
- CNN learning curves + test accuracy
- Retrieval with ML embeddings (CNN/PANNs/AST/CLAP)
- Zero-shot baselines (CLAP/Gemini)
