# Hallucination Mitigation Methods Can Increase Confidence in Private Attribute Inference

This repository evaluates whether hallucination-mitigation decoding methods make vision-language models safer when predicting sensitive human attributes.

Using **LLaVA-1.5-7B**, we compare standard decoding against three mitigation methods:

- **VCD** — Visual Contrastive Decoding
- **M3ID** — Multi-Modal Mutual-Information Decoding
- **ICD** — Instruction Contrastive Decoding

The experiments cover facial and contextual images across age, race, and gender prediction tasks.

## Main Finding

Hallucination mitigation did not consistently improve prediction accuracy. However, it increased model confidence in **17 of 18** method–dataset comparisons.

This suggests that methods designed to reduce hallucination can unintentionally strengthen confidence in sensitive-attribute inference, even when predictions do not become more accurate.

<p align="center">
  <img src="assets/figures/confidence_changes.png" width="900" alt="Confidence changes across experiments">
</p>

## Evaluation Pipeline

Each sensitive-attribute question is evaluated using cyclic answer-order permutations. Choice probabilities are aggregated to reduce answer-position bias.

<p align="center">
  <img src="assets/figures/evaluation_pipeline.jpg" width="900" alt="Evaluation pipeline">
</p>

The evaluation reports:

- Choice-based accuracy
- Mean choice confidence
- Expected Calibration Error with 10 bins
- Multiclass Brier score
- Continuity-corrected McNemar test for paired accuracy changes
- Wilcoxon signed-rank test for paired confidence changes

## Datasets

| Image type | Dataset | Attribute | Classes | Evaluation size |
|---|---|---:|---:|---:|
| Facial | UTKFace | Age | 5 | 3,000 |
| Facial | UTKFace | Race | 5 | 3,000 |
| Facial | FairFace | Gender | 2 | 3,000 |
| Facial | FairFace | Race | 7 | 2,925 |
| Context | PA-100K | Age | 3 | 3,000 |
| Context | MSP60K | Gender | 2 | 600 |

<p align="center">
  <img src="assets/figures/dataset_overview.png" width="850" alt="Dataset overview">
</p>

Raw dataset images are not distributed in this repository. Download each dataset from its official source and provide its local image directory when running an experiment.

Annotation files used by the experiments are provided under `data/`.

## Accuracy Results

| Dataset–Attribute | Baseline | VCD | M3ID | ICD |
|---|---:|---:|---:|---:|
| UTKFace–Age | 64.97 | 60.60 | 65.67 | 67.93 |
| PA-100K–Age | 45.13 | 46.03 | 45.67 | 43.70 |
| UTKFace–Race | 78.53 | 77.70 | 78.40 | 77.50 |
| FairFace–Race | 54.19 | 57.03 | 54.74 | 54.22 |
| FairFace–Gender | 94.53 | 94.57 | 94.83 | 94.67 |
| MSP60K–Gender | 58.50 | 63.67 | 61.00 | 58.17 |

Values are percentages. Accuracy improvements and reductions vary across datasets and decoding methods.

<p align="center">
  <img src="assets/figures/accuracy_results.png" width="900" alt="Accuracy results">
</p>

Calibration also changes inconsistently. Some settings show substantial ECE improvements, while others remain overconfident in specific confidence ranges.

<p align="center">
  <img src="assets/figures/reliability_diagrams.png" width="900" alt="Reliability diagrams">
</p>

Machine-readable final results are available in:

- `results/final_metrics.csv`
- `results/final_metrics.json`

## Repository Structure

```text
vlm-privacy/
├── assets/
│   └── figures/                 # Final figures used in this README
├── data/                        # Evaluation annotations
│   ├── fairface/
│   ├── msp60k/
│   ├── pa100k/
│   └── utkface_3k/
├── results/
│   ├── final_metrics.csv
│   └── final_metrics.json
├── scripts/
│   ├── analyze_results.py
│   ├── build_final_metrics.py
│   ├── build_utkface_annotations.py
│   └── validate_predictions.py
├── src/
│   ├── baseline_inference.py
│   ├── choice_scoring.py
│   ├── evaluate.py
│   ├── icd_inference.py
│   ├── m3id_inference.py
│   ├── model_loader.py
│   ├── vcd_core.py
│   └── vcd_inference.py
├── tests/
│   └── test_vcd_core.py
├── requirements.txt
└── run_experiment.py
```

## Installation

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default model is:

```text
llava-hf/llava-1.5-7b-hf
```

## Running an Experiment

The following example runs all four decoding methods on UTKFace age annotations:

```bash
python run_experiment.py \
  --annotations data/utkface/annotations_age.json \
  --image-dir /path/to/UTKFace/images \
  --results-dir results/utkface_age \
  --methods baseline vcd m3id icd \
  --resume
```

Replace `/path/to/UTKFace/images` with the local dataset directory.

Available methods:

```text
baseline vcd m3id icd
```

Useful options include:

```text
--limit N
--device DEVICE
--load-in-4bit
--resume
--overwrite
--seed 42
```

VCD parameters can be configured with:

```text
--vcd-alpha
--vcd-beta
--vcd-noise-step
--vcd-temperature
--vcd-top-p
--vcd-top-k
```

ICD parameters can be configured with:

```text
--icd-alpha
--icd-beta
--icd-temperature
--icd-top-p
--icd-top-k
--icd-disturbance-instruction
```

Each run produces local experiment files such as:

```text
pilot_predictions.jsonl
pilot_metrics.json
run_metadata.json
```

These large run-level files are intentionally excluded from Git.

## Rebuilding Final Metrics

After generating the expected local prediction folders, rebuild the consolidated result files with:

```bash
python scripts/build_final_metrics.py
```

This updates:

```text
results/final_metrics.csv
results/final_metrics.json
```

The consolidation script uses only examples that have paired predictions for Baseline, VCD, M3ID, and ICD.

## Limitations

This study evaluates multiple-choice sensitive-attribute inference with one VLM family. Results may differ for open-ended generation, other models, alternative prompts, and other privacy-sensitive attributes.

The experiments measure prediction behavior and confidence. They do not represent every form of real-world privacy leakage.

## Citation

Citation information will be added after publication.
