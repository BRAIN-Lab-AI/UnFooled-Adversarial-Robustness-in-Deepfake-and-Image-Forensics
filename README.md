# UnFooled: Attack-aware Deepfake Forensics

## Project Metadata
### Authors
- **Team:** Noor Fatima
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Deepfake and image-tampering detectors look great on clean test sets, but real-world photos don’t stay clean. People can re-save images to shift JPEG blocks, apply tiny warps to hide resampling traces, use AI fills to smooth seams, or run the file through social apps that strip useful camera noise. In low-light or surveillance footage (often compressed or infrared), these small edits can quietly break today’s detectors.

UnFooled tackles this by training the model to expect attacks. We:
1. Practice against a red team of common counter-forensics during training.
2. Combine content cues (what’s in the picture) with physics-like cues (noise patterns, resampling artifacts).
3. Use randomized checks at test time (slight crops/resizes/recompressions) and vote on the result.
The goal: a detector that stays accurate, well-calibrated, and explainable—even when the forger fights back.

## Problem Statement
We treat the task as (a) real vs. fake and (b) where is it fake (a heatmap), even after the image has been tweaked to fool us. Attackers may know our model (white-box), know our general tricks (gray-box), or only see outputs (black-box). They must keep changes hard to notice while keeping the edited content.

We assume messy “chain of custody” (e.g., WhatsApp/Telegram recompression) and surveillance quirks (rolling shutter, LED flicker, NIR).
Our questions:

Q1: Which counter-forensics hurt most, and by how much?

Q2: Does training with a mix of attacks improve worst-case robustness, not just average scores?

Q3: Do small random test-time jitters reduce attack transfer without being slow—and can we abstain when uncertain?

We will report drop in AUC under attack (ΔAUC), worst-case accuracy across attack types, and confidence calibration suitable for legal use.

## Application Area and Project Domain
Targets include law enforcement and media forensics. Users need: a clear real/fake score, a heatmap showing where the tamper likely is, and a confidence readout (with the option to abstain when unsure).

Our pipeline can also work with provenance standards (e.g., C2PA): if signed claims exist, we check them; if not, we rely on physics-style cues. This makes reports useful for internal reviews and courtroom exhibits.

## What is the paper trying to do, and what are you planning to do?
We propose UnFooled, an attack-aware detector that pairs red-team training with randomized test-time defense and two-stream features (content + residuals). During training, each batch is hit with the most damaging of several edits: JPEG re-align + recompress, tiny resampling warps, denoise→regrain (PRNU/noiseprint spoof), seam smoothing, small color/gamma shifts, and social-app transcodes. The model learns both to decide real/fake and to mark tampered pixels.

At test time, we run a few small random transforms (resize/crop phase, gamma tweak, JPEG quality/phase), get multiple predictions, and vote. Under the hood, we use a pretrained backbone (e.g., ResNet-50) plus a forensic residual adapter and a light FPN-style mask head—fast to fine-tune, sensitive to subtle traces. We will report clean vs. attacked metrics side-by-side (ΔAUC, worst-case accuracy, IoU for localization, and calibration/ECE) on standard deepfake/tamper datasets and a surveillance-style split (low-light, heavy compression). Success = small ΔAUC, strong worst-case, and clear, judge-friendly explanations—because a detector that only works when nobody’s trying to fool it isn’t very forensic.

## Project Documents
- **Presentation:** [Project Presentation](https://github.com/BRAIN-Lab-AI/UnFooled-Adversarial-Robustness-in-Deepfake-and-Image-Forensics/blob/main/noorDL.pptx)
- **Report:** [Project Report](https://github.com/BRAIN-Lab-AI/UnFooled-Adversarial-Robustness-in-Deepfake-and-Image-Forensics/blob/main/DL_PAPER.pdf)

### Reference Paper
- [Adversarial Threats to DeepFake Detection: A Practical Perspective](https://openaccess.thecvf.com/content/CVPR2021W/WMF/papers/Neekhara_Adversarial_Threats_to_DeepFake_Detection_A_Practical_Perspective_CVPRW_2021_paper.pdf)

### Reference Dataset
- [CIFAR-10](https://www.cs.toronto.edu/%7Ekriz/cifar.html)
- [DeepFakeFace Dataset](https://huggingface.co/datasets/OpenRL/DeepFakeFace)


## Project Technicalities

### Terminologies
* **Two-Stream Detector:** A model with a semantic/content stream and a forensic-residual stream; features are fused via a lightweight adapter for classification and evidence mapping. 
* **Residual stream:** Forensic branch fed by SRM/HPF (high-pass/Spatial Rich Model filters) + a small CNN to emphasize noise/edge artifacts rather than content.
* **SRM/HPF:** Fixed or learnable high-pass filters that suppress low-frequency content and reveal manipulation fingerprints. 
* **Lightweight Fusion (F):** Channel-gating + 1×1 mixing to merge content and residual features into a joint representation used by heads. 
* **Shallow FPN Evidence Head:** An upsampling head that aggregates multi-scale features to produce a tamper heatmap under weak supervision (face-region priors).
* **Mask head:** Lightweight conv stack (often 3×3 convs) that upsamples P-features to produce a tamper heatmap (HxW probabilities).
* **GAP (Global Average Pooling):** Averages each feature map channel over H×W to get one number per channel; used to form a compact vector for classification
* **Worst-of-K Red-Team Training:** For each image, sample multiple counter-forensic transforms and train on the one that maximally harms the classifier—hardening the model to realistic attacks (JPEG realign/recompress, subtle resampling, denoise→regrain/PRNU-spoof, seam smoothing, mild color/gamma shifts, social-app transcodes). 
* **Randomized Test-Time Augmentation (TTA):** Low-cost jitters (resize/crop phase, mild gamma, JPEG phase) with logit averaging for decisions and pixelwise max for heatmaps.
* **Weak Localization:** Training/evaluation of evidence maps using soft face-region priors instead of pixel-accurate masks; metrics emphasize “energy within ROI” and precision-in-ROI. 
* **Global Operating Point:** A single threshold chosen to maximize worst-case accuracy across clean and attacked splits, then fixed for reporting. 
* **Reliability Metrics:** Expected Calibration Error (ECE), Brier score, Negative Log-Likelihood (NLL), and selective prediction via AURC (risk–coverage). 
* **Regrain Stressor:** Denoise→regrain (PRNU spoof) is the hardest of the tested families but remains controlled with the combined training + TTA. 
## Problem Statements

* **Problem 1 — Robustness under realistic counter-forensics:** Detectors trained on clean or narrowly augmented data fail under recompression, resampling, regraining, and app transcodes common in the wild. 
* **Problem 2 — Reliability and calibration:** Reported confidence often misaligns with risk, and worst-case performance across attacks is rarely foregrounded. 
* **Problem 3 — Actionable evidence without dense masks:** Pixel-accurate tamper masks are scarce, yet analysts need spatial evidence that is legible and scalable. 

## Loopholes or Research Areas

* **Evaluation that matches deployment:** Standard clean-set metrics hide worst-case behavior; need stress protocols and worst-of-attack summaries. 
* **Weak-supervised localization:** Methods to produce reliable heatmaps using region priors instead of precise masks. 
* **Reliability diagnostics:** Routine use of ECE, NLL, Brier, and selective prediction (AURC) alongside AUC/accuracy. 
* **Phase-aware defenses:** Low-cost, non-differentiable jitters (resize/JPEG phase, mild gamma) tailored to forensic failure modes. 

## Problem vs. Ideation: Proposed Ideas

1. **Attack-Aware Training:** Worst-of-K mixture of realistic counter-forensics per mini-batch to harden features. 
2. **Phase-Aware TTA:** Randomized test-time jitters with aggregation to stabilize decisions and calibration. 
3. **Two-Stream + Shallow FPN:** Fuse semantic content and residual cues via a lightweight adapter; emit weakly supervised heatmaps for evidence.

<img width="1781" height="1106" alt="unfooled drawio" src="https://github.com/user-attachments/assets/4901da73-f4e8-494a-a27e-95f2a2bcf12a" />

# Proposed Solution: Code-Based Implementation

* **Modified Architecture:** Pretrained content backbone + residual extractor/encoder; lightweight fusion; classifier head; shallow FPN mask head. 
* **Training Regimen:** Deterministic preprocessing; worst-of-K red-team transforms with weighted BCE + soft-Dice for masks, edge/size regularizers, and cross-view consistency. 
* **Inference:** Randomized TTA with logit averaging and heatmap max-pool aggregation; single global operating point.
* **Evaluation:** Clean vs attacked counterparts, worst-case accuracy, calibration (ECE/NLL/Brier), and weak-localization summaries (energy/precision in ROI).
* 
<img width="899" height="194" alt="pipeline" src="https://github.com/user-attachments/assets/96378125-c14d-45f1-b3c8-f6aa733ddda6" />

# Key Components

* `model.py` — Content encoder, residual extractor/encoder, fusion adapter, classifier head, shallow FPN evidence head. 
* `train.py` — Worst-of-K red-team pipeline, loss stack (wBCE + Dice + edge/size + consistency), mixed precision, deterministic seeds. 
* `utils.py` — Preprocessing Π(·), attack operators (jpeg/warp/regrain/seam/gamma/transcode) with fixed ranges, face-prior generation (InsightFace) for weak supervision. 
* `inference.py` — Randomized TTA (N views), probability/logit aggregation, evidence heatmap max-pool, global thresholding. 

# Model Workflow

## Input

* **Face Thumbnail(s):** Real/fake images (held-out identities; optionally a surveillance-style split). 
* **Preprocessing Π(·):** Color/dynamic-range standardization; resize to fixed resolution. 
* **Weak Prior (g):** Soft face-region mask (expanded box + Gaussian) for evidence supervision/evaluation. 


## Diffusion-Style “Refinement” Analogue (in our detector)

* **Residual Path:** Apply (R(·)) (high-pass/SRM/wavelet) → residual encoder → features. 
* **Content Path:** Pretrained backbone → semantic features. 
* **Fusion + Heads:** Adapter → classifier logit (s) and FPN mask-logits (z). Evidence head upsamples multi-scale features to an input-aligned heatmap. 

## Training (Attack-Aware)

* **Worst-of-K:** For each sample, pick the most damaging transform among the sampled K edits and train on that view; include clean-view mask consistency.

<img width="2370" height="1193" alt="1" src="https://github.com/user-attachments/assets/3a892ae0-8c0e-4632-b815-071fd358a945" />


## Inference (Deployment-Facing)

* **Randomized TTA:** Apply small jitters (crop/resize phase, mild gamma, JPEG phase); average logits for probability; take pixelwise max over heatmaps to preserve localized peaks. 
* **Outputs:** (1) Decision + calibrated confidence; (2) Aggregated evidence heatmap concentrated within plausible face regions.
<img width="2766" height="2392" alt="55" src="https://github.com/user-attachments/assets/c82a1525-627d-45be-b558-f8934dbe770b" />

## How to Run the Code

### 1) Get the sources

**Option A — GitHub**

```bash
git clone https://github.com/BRAIN-Lab-AI/UnFooled-Adversarial-Robustness-in-Deepfake-and-Image-Forensics
cd unfooled_code
```

**Option B — Download ZIP**

* Download ZIP

### 2) Set up the environment

```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Quick check:

```bash
python -c "import unfooled, sys; print('unfooled', unfooled.__version__); print(sys.version)"
```

### 3) Train

Use the provided CLI entry point (recommended):

```bash
# minimal example
unfooled-train --data /path/to/data --epochs 1 --batch-size 8

# typical GPU run
unfooled-train --data /path/to/data --epochs 50 --batch-size 32 --device cuda
```

Or run the script directly:

```bash
python scripts/train.py --data /path/to/data --epochs 50 --batch-size 32 --device cuda
```

> Wire your actual training loop inside `src/unfooled/training/train.py` (the entry points call into that module).

### 4) Inference / Evaluation

Using the CLI:

```bash
unfooled-eval --data /path/to/data --checkpoint /path/to/checkpoint.pt --device cuda
```

Or via script:

```bash
python scripts/eval.py --data /path/to/data --checkpoint /path/to/checkpoint.pt --device cuda
```


### Notes

* **Data paths:** point `--data` to your prepared dataset root. Add/modify loaders in `src/unfooled/data/datasets.py`.
* **Checkpoints:** training scripts should save `.pt` files; pass one to `--checkpoint` for evaluation.
* **Devices:** use `--device cuda` for GPU, `--device cpu` for CPU.
* **Editable install:** `pip install -e .` lets you edit code in `src/unfooled/` without reinstalling.


## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
