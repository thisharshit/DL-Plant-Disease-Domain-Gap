# Measuring and Closing the Domain Gap in Deep Learning Plant Disease Diagnosis

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Vision%20Transformer-ViT--B%2F16-blue.svg)](https://arxiv.org/abs/2010.11929)
[![Report PDF](https://img.shields.io/badge/Project%20Report-PDF-red.svg?logo=adobeacrobatreader&logoColor=white)](report/DL_Project_P05_Report.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deep Learning Final Project (P05) - Sitare University**  
> **Team 5:**
> - **Harshit Chaturvedi** (`harshitvijay2175@gmail.com`)
> - **Ajay Jatav** (`jatavaj09@gmail.com`)
> - **Harsh Chourasiya** (`harshchourasiya295@gmail.com`)

---

## Team Work Distribution & Roles

| Member | Primary Role | Key Deliverables & Modules Owned |
| :--- | :--- | :--- |
| **Harshit Chaturvedi** | Lead Architect & Modeling | - Project pipeline design & PyTorch model factory (`src/models.py`)<br>- Vision Transformer (ViT-B/16) and ResNet-50 integration<br>- Colab notebook orchestration & final paper write-up |
| **Ajay Jatav** | Data Engineering & Preprocessing | - PlantVillage & PlantDoc dataset curation and class matching (`src/dataset.py`)<br>- 8-class cross-domain alignment dictionary<br>- Heavy field-simulating augmentation pipeline (color jitter, blur, cutouts) |
| **Harsh Chourasiya** | Domain Adaptation & Analysis | - HSV color segmentation and background masking (`src/bg_removal.py`)<br>- Few-shot field fine-tuning experiments (`src/few_shot_adapt.py`)<br>- Cross-domain evaluation metrics, F1-scores, and benchmark synthesis (`src/evaluate.py`) |

---

## Problem Overview
Deep learning models trained on sanitized, laboratory-controlled leaf datasets such as **PlantVillage** routinely achieve near-perfect classification accuracies (>98%). However, when evaluated on real-world in-the-wild agricultural field photographs (**PlantDoc**), these models suffer a catastrophic performance degradation, frequently dropping below 40% accuracy. 

This failure stems from a severe **Domain Shift**:
1. **Background Artifacts:** Clean, uniform laboratory backgrounds vs. soil, weeds, foliage clutter, and non-target leaves.
2. **Illumination & Shadows:** Constant diffuse lighting vs. harsh Indian direct sunlight, occlusions, and dynamic shadows.
3. **Multi-Scale Symptoms:** Single centered leaves vs. complex multi-leaf clusters captured at varying focal distances on budget smartphones.

This project **quantifies this domain gap** and implements four systematic interventions to close it:
- Heavy field-simulating data augmentations.
- Background removal and morphological lesion segmentation.
- Few-shot target domain fine-tuning ($k$-shot field adaptation).
- Vision Transformer (ViT-B/16) comparative analysis.

---

## Cross-Domain Benchmark Results

Experiments conducted across aligned crop disease classes (Tomato Early/Late Blight, Bacterial Spot, Septoria, Potato Blights, Apple Scab, Bell Pepper Spot) evaluated on authentic PlantDoc field conditions:

| Method / Architecture | In-Domain Acc (Lab %) | Cross-Domain Acc (Field %) | Domain Gap (Delta pp) | Field Macro-F1 (%) | Retention Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 (Edge Baseline)** | 86.46% | 27.03% | 59.43 pp | 23.87% | 31.3% |
| **ResNet-50 (Baseline)** | 92.19% | 24.32% | **67.86 pp** | 17.26% | 26.4% |
| **ResNet-50 + Heavy Augmentation** | 85.94% | 33.78% | **52.15 pp** | 29.75% | 39.3% |
| **ResNet-50 + Background Masking** | 13.02% | 18.92% | -5.90 pp | 11.45% | 145.3% |
| **ResNet-50 + Few-Shot Adaptation (10-shot)** | 84.90% | **37.84%** | **47.06 pp** | **35.74%** | **44.6%** |
| **Vision Transformer (ViT-B/16 Baseline)** | 64.06% | 24.32% | 39.74 pp | 20.64% | 38.0% |
| **ViT-B/16 + Heavy Augmentation + Few-Shot** | 33.33% | 35.14% | -1.80 pp | 30.73% | 105.4% |

### Key Takeaways
1. **Severe Domain Shift Validated:** The baseline ResNet-50 experiences a catastrophic **67.86 percentage points accuracy drop** when transferring from controlled laboratory conditions (PlantVillage) to realistic outdoor field photos (PlantDoc), collapsing from 92.19% to 24.32% (retaining only 26.4% of its capability).
2. **Few-Shot Adaptation Achieves Peak Field Accuracy:** Fine-tuning on merely 10 field images per class achieves the highest field diagnostic accuracy of **37.84%** and the highest Field Macro-F1 of **35.74%**, closing the domain gap to 47.06 pp and achieving a retention rate of 44.6%.
3. **Heavy Data Augmentation Closes the Gap Without Field Labels:** Simulating outdoor illumination, shadows, and blur lifts field accuracy from 24.32% to **33.78%** (+9.46 pp gain) and elevates Macro-F1 from 17.26% to **29.75%**, reducing the domain gap to 52.15 pp without requiring any target field annotations.
4. **ViT Generalization & Adaptation:** Pretrained Vision Transformers without adaptation score 24.32% on field data, but combining Heavy Augmentation with Few-Shot adaptation boosts field accuracy to **35.14%** (Macro-F1: **30.73%**).
5. **Edge Efficiency (MobileNetV2):** MobileNetV2 achieves 27.03% field accuracy (outperforming unadapted ResNet-50) with only 3.5M parameters (14 MB), validating its practical utility for budget Android smartphones in Indian agriculture.

---

## Repository Structure
```
DL/
|-- notebooks/
|   \-- DL_Project_P05_Domain_Gap_Colab.ipynb  # Interactive Google Colab Notebook
|-- src/
|   |-- __init__.py
|   |-- dataset.py        # Dataset mapping, loaders, and augmentation pipelines
|   |-- models.py         # ResNet-50, MobileNetV2, and ViT-B/16 architectures
|   |-- train.py          # Training loop with validation and checkpoints
|   |-- evaluate.py       # Cross-domain evaluator, domain gap metric, and tables
|   |-- bg_removal.py     # HSV leaf lesion segmentation and background masking
|   \-- few_shot_adapt.py # Few-shot target domain adaptation
|-- report/
|   |-- DL_Project_P05_Report.pdf # Final Compiled Research Paper (PDF)
|   \-- Elcvia2e.tex              # LaTeX Research Report Source
|-- requirements.txt      # Python dependencies
\-- README.md             # Project documentation
```

---

## Quickstart & Execution

### 1. Run on Google Colab (Recommended)
Open `notebooks/DL_Project_P05_Domain_Gap_Colab.ipynb` directly in Google Colab with free T4 GPU. Run all cells sequentially to download datasets, train models, and generate benchmark tables and plots.

### 2. Local Setup
```bash
# Clone the repository
git clone https://github.com/thisharshit/DL-Plant-Disease-Domain-Gap.git
cd DL

# Install dependencies
pip install -r requirements.txt

# Run full cross-domain training & evaluation
python -m src.train
```

---

## References
1. **Mohanty et al. (2016)**: *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science. [arXiv:1604.03169](https://arxiv.org/abs/1604.03169)
2. **Singh et al. (2020)**: *PlantDoc: A Dataset for Visual Plant Disease Detection in Natural Field Conditions*, ACM CoDS-COMAD. [arXiv:1911.10317](https://arxiv.org/abs/1911.10317)
3. **Dosovitskiy et al. (2020)**: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
