# Hallucination Mitigation Can Increase Confidence in Private Attribute Inference

This repository investigates whether hallucination-mitigation decoding methods make vision-language models safer when inferring privacy-sensitive demographic attributes.

We evaluate **LLaVA-1.5-7B** with four decoding methods:

- **Baseline**
- **VCD** — Visual Contrastive Decoding
- **M3ID** — Multi-Modal Mutual-Information Decoding
- **ICD** — Instruction Contrastive Decoding

The experiments cover age, gender, and race inference using facial and pedestrian images.

## Project Context

This work was conducted during the **2026 USTC Future Scientist Student Exchange Program (FUSEP)** at the University of Science and Technology of China.

## Research Poster

<p align="center">
  <a href="assets/poster/research_poster.png">
    <img
      src="assets/poster/research_poster.png"
      width="440"
      alt="Research poster"
    >
  </a>
</p>

<p align="center">
  <sub>
    Research poster prepared for the 2026 USTC Future Scientist Student Exchange Program.
  </sub>
</p>

## Key Finding

> Hallucination mitigation increased confidence in **17 of 18 comparisons**, while its effects on accuracy and calibration were inconsistent.

The only confidence decrease was observed for ICD on FairFace–Gender. These results suggest that stronger visual grounding can make sensitive-attribute predictions appear more reliable without consistently making them more accurate.

<p align="center">
  <img
    src="assets/figures/confidence_changes.png"
    width="720"
    alt="Changes in confidence from the baseline"
  >
</p>

## Evaluation Pipeline

Each image is evaluated using multiple-choice sensitive-attribute questions.

To reduce answer-position bias, the order of the answer choices is cyclically permuted. Choice probabilities are then aggregated across permutations to produce the final prediction and confidence score.

<p align="center">
  <img
    src="assets/figures/evaluation_pipeline.jpg"
    width="360"
    alt="Evaluation pipeline"
  >
</p>

The evaluation reports:

- Accuracy
- Mean choice confidence
- Expected Calibration Error (ECE)
- Multiclass Brier score
- Continuity-corrected McNemar test for accuracy
- Wilcoxon signed-rank test for confidence

## Evaluation Settings

<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Image type</th>
      <th align="center">Dataset</th>
      <th align="center">Attribute</th>
      <th align="center">Classes</th>
      <th align="center">Samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Facial</td>
      <td align="center">UTKFace</td>
      <td align="center">Age</td>
      <td align="center">5</td>
      <td align="center">3,000</td>
    </tr>
    <tr>
      <td align="center">Facial</td>
      <td align="center">UTKFace</td>
      <td align="center">Race</td>
      <td align="center">5</td>
      <td align="center">3,000</td>
    </tr>
    <tr>
      <td align="center">Facial</td>
      <td align="center">FairFace</td>
      <td align="center">Gender</td>
      <td align="center">2</td>
      <td align="center">3,000</td>
    </tr>
    <tr>
      <td align="center">Facial</td>
      <td align="center">FairFace</td>
      <td align="center">Race</td>
      <td align="center">7</td>
      <td align="center">2,925</td>
    </tr>
    <tr>
      <td align="center">Pedestrian</td>
      <td align="center">PA-100K</td>
      <td align="center">Age</td>
      <td align="center">3</td>
      <td align="center">3,000</td>
    </tr>
    <tr>
      <td align="center">Pedestrian</td>
      <td align="center">MSP60K</td>
      <td align="center">Gender</td>
      <td align="center">2</td>
      <td align="center">600</td>
    </tr>
  </tbody>
</table>

</div>

<p align="center">
  <img
    src="assets/figures/dataset_overview.png"
    width="680"
    alt="Dataset overview"
  >
</p>

Raw dataset images are not included in this repository. The evaluation annotation files are provided under `data/`.

## Results

Accuracy changes varied across datasets and methods. Some methods significantly improved accuracy, while others significantly reduced it.

<p align="center">
  <img
    src="assets/figures/accuracy_results.png"
    width="650"
    alt="Accuracy results"
  >
</p>

Calibration also changed inconsistently. For example, VCD substantially improved calibration on MSP60K–Gender, but calibration did not improve uniformly across all settings.

<p align="center">
  <img
    src="assets/figures/reliability_diagrams.png"
    width="700"
    alt="Representative reliability diagrams"
  >
</p>

The consolidated results are available in machine-readable formats:

- [`results/final_metrics.csv`](results/final_metrics.csv)
- [`results/final_metrics.json`](results/final_metrics.json)

## Repository Structure

```text
vlm-privacy/
├── assets/
│   ├── figures/                 # Figures used in this README
│   └── poster/
│       └── research_poster.png
├── data/
│   ├── fairface/                # Gender and race annotations
│   ├── msp60k/                  # Gender annotations
│   ├── pa100k/                  # Age annotations
│   └── utkface/                 # Age and race annotations
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

Create a virtual environment and install the required packages:

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

The following example evaluates all four methods on UTKFace age inference:

```bash
python run_experiment.py \
  --annotations data/utkface/annotations_age.json \
  --image-dir /path/to/UTKFace/images \
  --results-dir results/utkface_age \
  --methods baseline vcd m3id icd \
  --resume
```

Replace `/path/to/UTKFace/images` with the local path to the dataset images.

Available methods:

```text
baseline vcd m3id icd
```

Useful options:

```text
--limit N
--device DEVICE
--load-in-4bit
--resume
--overwrite
--seed 42
```

Run-level prediction files and metadata are stored locally and excluded from Git.

## Testing

Run the VCD core tests with:

```bash
python -m pytest tests
```

## Rebuilding Final Metrics

The repository includes the consolidated final metrics, but run-level prediction files are excluded from Git. To regenerate the metrics from scratch, first complete the expected experiment runs and then execute:

```bash
python scripts/build_final_metrics.py
```

This updates:

```text
results/final_metrics.csv
results/final_metrics.json
```

The script uses samples with paired predictions from Baseline, VCD, M3ID, and ICD.

## Data Availability

Raw dataset images are not distributed in this repository. Users must obtain UTKFace, FairFace, PA-100K, and MSP60K from their official sources and comply with each dataset's license and terms of use.

The files under `data/` provide the evaluation annotations used in this study.

## Responsible Use

This repository is intended for research on privacy risks and vision-language model behavior. It should not be used to infer sensitive attributes about individuals for surveillance, profiling, access control, employment, healthcare, or other real-world decision-making.

## Limitations

This study evaluates one vision-language model family and multiple-choice demographic-attribute inference.

Results may differ for:

- Other VLMs
- Open-ended generation
- Different prompt formats
- Indirect private attributes
- Privacy-related objects
- Other demographic or contextual attributes

## References

1. Liu et al. **Improved Baselines with Visual Instruction Tuning.** CVPR, 2024.
2. Leng et al. **Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding.** CVPR, 2024.
3. Favero et al. **Multi-Modal Hallucination Control by Visual Information Grounding.** CVPR, 2024.
4. Wang et al. **Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding.** Findings of ACL, 2024.
5. Zhang et al. **Age Progression/Regression by Conditional Adversarial Autoencoder.** CVPR, 2017.
6. Kärkkäinen and Joo. **FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation.** WACV, 2021.
7. Liu et al. **HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis.** ICCV, 2017.
8. Jin et al. **Pedestrian Attribute Recognition: A New Benchmark Dataset and a Large Language Model Augmented Framework.** AAAI, 2025.
