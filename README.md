# ReXTrust: XAI-Driven Multi-Disease Diagnosis in Chest X-rays via Conditional Enhancement and Grounded Reporting

## Overview

ReXTrust is an end-to-end, explainable AI pipeline for multi-disease diagnosis from chest X-rays. It addresses three unresolved challenges in AI-based chest X-ray diagnosis:

1. **Degraded image quality** — X-rays from low-resource settings often suffer from noise, low contrast, and poor illumination.
2. **Lack of per-disease spatial verification** — existing explainability methods produce a single blended saliency map instead of independently verifiable, per-disease heatmaps.
3. **Ungrounded AI-generated reports** — vision-language models can generate fluent but hallucinated clinical text with no verification against external medical knowledge.

The system is built as a five-stage pipeline:

```
Chest X-ray → Image Enhancement → Disease Detection → Explainability (Score-CAM)
           → Report Generation (MediVLM) → Grounding & Validation (RadCheck RAG)
```

---

## Pipeline Architecture

1. **Input** — Chest X-ray from the CheXpert dataset (DICOM/JPEG).
2. **Image Quality Assessment** — Routes each image to enhancement if quality is low, or passes it unchanged if acceptable.
3. **Image Enhancement** — Attention U-Net + CLAHE improves contrast and denoises without destroying anatomical/clinical detail.
4. **Disease Detection** — Multi-label classifier producing per-disease confidence scores.
5. **Explainability** — Score-CAM generates an independent saliency heatmap per predicted disease.
6. **Report Generation** — MediVLM (vision-language model) generates symptom/clinical text grounded in the detected findings.
7. **RadCheck — Report Grounding** — Validates the generated report against external medical evidence and internal consistency, producing a confidence-scored, hallucination-checked report.
8. **Structured Diagnostic Output** — Enhanced image + disease confidence scores + per-disease saliency maps + RAG-grounded clinical report.

---

## Module 1: Image Enhancement

**Objective:** Improve visual quality and contrast of chest X-rays while preserving anatomical detail needed by downstream detection and reporting stages.

### Methods Implemented & Compared

**Classical & learned baselines**
- **CLAHE** — clip-limited adaptive histogram equalization (classical baseline)
- **UniMIE** — pretrained diffusion-prior backbone (Fei et al., 2025)
- **Zero-DCE** — unsupervised DCE-Net with spatial-consistency, exposure, color-constancy, and illumination-smoothness losses (Guo et al., 2020)
- **U-Net** — supervised model trained on CLAHE-target pairs with L1 + SSIM + Laplacian loss, 60 epochs (Ronneberger et al., 2015)

**Proposed hybrid pipelines**
- **U-Net + CLAHE** — U-Net denoises, CLAHE boosts contrast on the clean output
- **Attention U-Net + CBAM** — 4-stage ResBlock encoder/decoder with CBAM channel+spatial attention at the bottleneck and Attention Gates on every skip connection
- **Attention U-Net + CLAHE (Ours — final proposed pipeline)** — best average rank across all 5 evaluation metrics

### Proposed Architecture: Attention U-Net + CLAHE

- **Residual Blocks** — preserve important features while improving gradient/feature flow
- **CBAM** — channel + spatial attention applied at the bottleneck
- **Attention Gates** — selectively pass relevant encoder features to the decoder via skip connections
- **CLAHE (post-processing)** — enhances local contrast (clip limit 2.0, tile size 8×8) while limiting noise amplification

### Evaluation

Evaluated on 23 low-quality CheXpert images across 8 methods using **BRISQUE (↓), Entropy (↑), Sharpness (↑), CII (↑), EME (↑)**.

| Method | BRISQUE ↓ | Entropy ↑ | Sharpness ↑ | CII | Avg Rank |
|---|---|---|---|---|---|
| **Attn U-Net + CLAHE (Ours)** | 13.35 | **7.92** | 2082.3 | **1.90** | **2.2** |
| U-Net V3 + CLAHE | 17.51 | 7.91 | 1859.7 | 1.76 | 3.2 |
| CLAHE | 7.01 | 7.84 | 3609.6 | 1.71 | 3.8 |
| Attention U-Net | 13.22 | 7.86 | 1295.7 | 1.37 | 4.6 |
| Original | 5.62 | 7.85 | 534.3 | 1.00 | 5.2 |
| Zero-DCE | 19.59 | 6.54 | 1766.4 | 1.27 | 5.4 |
| U-Net V3 | 17.96 | 7.86 | 1190.9 | 1.29 | 5.6 |
| UniMIE | -0.73 | 7.85 | 725.5 | 0.93 | 6.0 |

**Key takeaways:**
- Best average rank (2.2) across all 5 metrics on 23 CheXpert test images.
- Highest Entropy (7.92) → retains the most diagnostically-relevant structural detail.
- Highest CII (1.90) → best contrast improvement, clearer anatomical boundaries.
- Stays in the top 2 methods on 4 of 5 metrics.
- Support devices (wires/tubes/leads) remain visible after enhancement — quality improves without erasing clinically relevant structural detail.

---

## Module 2: Disease Detection

**Objective:** Predict Cardiomegaly, Edema, and Pleural Effusion from chest X-rays, and compare architectures under one identical, leak-free data/evaluation protocol so gains are attributable to model design, not data variation.

### Shared Data Foundation (built once, reused by all techniques)

```
Raw CheXpert Data → Balanced Subsample (frontal only, per-disease sampling)
                  → Patient-Level Split (70/15/15, zero leakage)
                  → Shared Loaders (Train / Val / Test)
```

All four techniques train, validate, and test on the exact same images with thresholds tuned only on the validation split, so architecture is the only variable being compared.

### Techniques Compared

1. **CNN (Baseline Recipe)** — DenseNet121 → GAP → Linear(3). Plain BCE loss, no dropout, no scheduler (literal CheXNet recipe, literature-comparable reference).
2. **CNN (Matched Recipe)** — Same architecture as baseline, with Focal Loss + dropout (0.3) + LR scheduler; isolates the "recipe" effect.
3. **Hybrid CNN-Transformer** — DenseNet121 features → patch tokens + [CLS] → 2-layer Transformer encoder → classify off [CLS]. Uses the matched recipe.
4. **CheXGCN** — Image features + a GCN over the disease co-occurrence graph (built from the train split only), dot-product classifier. Uses the matched recipe.

Thresholds are tuned per disease on the validation set only, then frozen and applied to the test set for Precision/Recall/F1.

### Evaluation Across Data Scales (1k, 6k, 20k images per disease)

**Key takeaways:**
- **CheXGCN** shows the largest improvement with increased data — lowest AUC at 1k, tied-highest AUC on Cardiomegaly and Pleural Effusion at 20k, consistent with the graph-based co-occurrence signal needing more samples to be reliable.
- **Most performance gains occur between 1k and 6k images**, with a plateau (and occasional slight decrease) from 6k to 20k — diminishing returns beyond 6k.
- **Hybrid CNN-Transformer** needs more data to be competitive — lowest/near-lowest F1/AUC at 1k, consistent with higher data requirements of Transformer attention vs. the built-in spatial inductive bias of CNNs.
- **Relative rankings shift substantially with scale** — at 6k/20k, no single architecture dominates; Hybrid and CheXGCN each top at least one disease while the CNN baseline drops to 2nd/3rd on several occasions. Architecture choice matters less than data volume at this scale.

---

## Module 3: Explainability — Score-CAM

```
Conv Feature Maps → Upsample & Normalize → Mask Input & Forward Pass
                  → Weighted Sum + ReLU → Upsampled Heatmap
```

- **Why Score-CAM:** Gradient-free — instead of backpropagating gradients (as in Grad-CAM), each activation map is used directly as an image mask and scored via a forward pass, avoiding gradient noise/saturation in deep CNNs.
- **Target layer:** Final DenseNet121 conv block (`denseblock4` / `norm5`) — deep enough to encode disease-relevant spatial features, coarse enough to localize well after upsampling.
- **Independent per disease:** Run separately for each of the 3 diseases using that disease's own class score, so each heatmap is independently, spatially verifiable — not one blended saliency map.
- CheXGCN was selected for the clearest per-disease spatial differentiation, independent of raw AUC/F1 ranking.

---

## Module 4: Report Generation — MediVLM

**Objective:** Generate a radiology report from a chest X-ray using region-aware visual understanding combined with medical language representation.

### Visual Understanding

```
Chest X-ray → Faster R-CNN (ResNet-34) → Top-p Anatomical Region Patches
           → CLIP ViT-L/14 (Visual Features) + Normalized Box (x,y,w,h) → MLP Position Encoding
           → Visual tokens (appearance + spatial information)
```

1. **Detect regions** — Faster R-CNN proposes salient anatomical regions via bounding boxes (region extraction, not disease classification).
2. **Select patches** — regions ranked by confidence; top salient patches used by default.
3. **Add spatial context** — CLIP ViT-L/14 extracts visual features; normalized box coordinates retain where each region occurs.

### Training and Inference

- **Training:** Chest X-ray → Faster R-CNN + CLIP → Visual Features → Contrastive Alignment with reference report (via ClinicalBERT) → Cross-Attention → GPT-2 → Report.
- **Inference:** No reference report available; new X-ray → visual features → decoder generates the report directly.

### Datasets & Training Setup

| Dataset | Train | Val | Test |
|---|---|---|---|
| MIMIC-CXR | 369K | 3.0K | 5.2K |

Training: AdamW optimizer, LR = 2 × 10⁻⁵, batch size 32, 30–50 epochs.

### Evaluation Metrics
- **NLG:** BLEU-1 to BLEU-4, METEOR, ROUGE-L
- **Semantic/Clinical:** BERTScore, RadGraph-F1, RaTEScore
- **Additional:** Severity Score

### Our Report-Generation Results

| Metric | GPT-2 | Qwen |
|---|---|---|
| BLEU-1 | 0.3631 | 0.2111 |
| BLEU-2 | 0.2401 | 0.1444 |
| BLEU-3 | 0.1707 | 0.1061 |
| BLEU-4 | 0.1228 | 0.0784 |
| METEOR | 0.3700 | 0.3456 |
| ROUGE-L | 0.2700 | 0.2237 |

GPT-2 is the original MediVLM decoder; Qwen is an experimental alternative decoder. GPT-2 scores higher on BLEU-1-4, METEOR, and ROUGE-L in this experiment; Qwen additionally reports a BERTScore of 0.7864.

**Why this architecture:** Selective salient patching improves performance over whole-image processing, while ClinicalBERT gives stronger text representation than the alternative CLIP text encoder.

---

## Module 5: RadCheck RAG — Clinical Hallucination Detection

**Why it's needed:** LLM-generated radiology reports can hallucinate findings not actually present in the image — sounding fluent but being clinically wrong. RadCheck RAG detects this.

**Core idea:** Hallucination detection combines external evidence (is the finding medically documented?) with internal consistency checks (does it contradict another finding in the same report?).

### Previous Architecture

```
Report → Clinical Claims Extraction → Construct MKG (Medical Knowledge Graph)
       → DFS Traversal + Confidence Propagation → PubMed Grounding → Trust Score → Final Verdict
```

**Drawbacks:**
- No explicit contradiction detection — only confidence propagation via BFS
- PubMed grounding applied uniformly rather than selectively
- No bidirectional relationship modeling between claims and KG nodes
- Lacked a structured typical-vs-atypical decision path, increasing unnecessary computation

### Current Architecture

```
Report → Clinical Claims Extraction → LLM Baseline Check (typical / atypical)
       → [if atypical] PubMed Grounding → MKG Contradiction Engine
       → Final Verdict: Hallucinated / Not Hallucinated (with confidence score)
```

**Pipeline stages:**
1. **Clinical Claims Extraction** — extracts discrete claims (target + status) from the report.
2. **LLM Baseline Check** — flags each claim as typical/atypical.
3. **PubMed Grounding** — invoked only for atypical findings, reducing unnecessary computation.
4. **MKG Contradiction Engine** — builds a Medical Knowledge Graph from the report's claims and runs contradiction detection.
5. **Final Verdict** — per-claim PASS/FAIL with confidence score, plus an overall report-level verdict (e.g., "REPORT IS CLINICALLY CONSISTENT").

### Architecture Comparison (System A vs. System B)

| Metric | System A | System B |
|---|---|---|
| Accuracy | 86.96% | 73.91% |
| Precision | 91.67% | 76.92% |
| Recall | 84.62% | 76.92% |
| F1-score | 88.00% | 76.92% |
| Specificity | 90.00% | 70.00% |

*Note: Evaluation was performed on a curated set of manually labeled clinical statements rather than raw dataset reports, since MIMIC-CXR/IU-Xray reports have no ground-truth contradiction/hallucination labels needed to compute precision, recall, and F1.*

---

## Technical Enrichment — Courses (Ongoing)

- **AI for Medical Diagnosis** — Coursera
- **Generative AI with LLMs** — Coursera
- **End-to-End Multimodal AI: Fine-Tuning, Fusion & MLOps** — Coursera

---

## References

1. **UniMIE:** Fei et al., *Communications Medicine*, 2025.
2. **Zero-DCE:** Guo et al., *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
3. **U-Net:** Ronneberger et al., *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015.
4. **MediVLM:** MediVLM: A Vision Language Model for Radiology Report Generation from Medical Images.
5. **Agentic MKG:** Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge.
6. **CheXpert:** Irvin et al., *AAAI Conference on Artificial Intelligence*, 2019.
7. **Score-CAM:** Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks.

---

## Project Status

- Image Enhancement, Disease Detection, Explainability (Score-CAM), MediVLM report generation, and RadCheck RAG grounding modules have each been developed and evaluated individually.
- Current focus: integrating the Image Enhancement module into the complete end-to-end pipeline and finalizing the shared evaluation protocol across all Disease Detection techniques.
