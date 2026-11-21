# Measuring and Closing the Domain Gap in Deep Learning Plant Disease Diagnosis

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Vision%20Transformer-ViT--B%2F16-blue.svg)](https://arxiv.org/abs/2010.11929)
[![Report PDF](https://img.shields.io/badge/Project%20Report-PDF-red.svg?logo=adobeacrobatreader&logoColor=white)](report/DL_Project_P05_Report.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deep Learning Final Project (P05) — Sitare University**  
> **Team 5:**
> - **Harshit Chaturvedi** (`harshitvijay2175@gmail.com`)
> - **Ajay Jatav** (`jatavaj09@gmail.com`)
> - **Harsh Chourasiyaa** (`harshchourasiya295@gmail.com`)

---

## 👥 Team Work Distribution & Roles

| Member | Primary Role | Key Deliverables & Modules Owned |
| :--- | :--- | :--- |
| **Harshit Chaturvedi** | Lead Architect & Modeling | • Project pipeline design & PyTorch model factory (`src/models.py`)<br>• Vision Transformer (ViT-B/16) and ResNet-50 integration<br>• Colab notebook orchestration & final paper write-up |
| **Ajay Jatav** | Data Engineering & Preprocessing | • PlantVillage & PlantDoc dataset curation and class matching (`src/dataset.py`)<br>• 8-class cross-domain alignment dictionary<br>• Heavy field-simulating augmentation pipeline (color jitter, blur, cutouts) |
| **Harsh Chourasiya** | Domain Adaptation & Analysis | • HSV color segmentation and background masking (`src/bg_removal.py`)<br>• Few-shot field fine-tuning experiments (`src/few_shot_adapt.py`)<br>• Cross-domain evaluation metrics, F1-scores, and benchmark synthesis (`src/evaluate.py`) |

---

## 📌 Problem Overview
Deep learning models trained on sanitized, laboratory-controlled leaf datasets such as **PlantVillage** routinely achieve near-perfect classification accuracies ($>98\%$). However, when evaluated on real-world in-the-wild agricultural field photographs (**PlantDoc**), these models suffer a catastrophic performance degradation, frequently dropping below $40\%$ accuracy. 

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

## 📊 Cross-Domain Benchmark Results

Experiments conducted across aligned crop disease classes (Tomato Early/Late Blight, Bacterial Spot, Septoria, Potato Blights, Apple Scab, Bell Pepper Spot):

| Method / Architecture | In-Domain Acc (Lab %) | Cross-Domain Acc (Field %) | Domain Gap ($\Delta$ %) | Field Macro-F1 (%) | Retention Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Baseline)** | 98.42% | 38.15% | **60.27%** | 34.20% | 38.8% |
| **MobileNetV2 (Edge Baseline)** | 97.10% | 35.40% | **61.70%** | 31.80% | 36.5% |
| **ResNet-50 + Heavy Augmentation** | 96.85% | 52.30% | 44.55% | 49.60% | 54.0% |
| **ResNet-50 + Background Masking** | 95.20% | 47.90% | 47.30% | 44.50% | 50.3% |
| **ResNet-50 + Few-Shot Adaptation (10-shot)** | 97.15% | 68.45% | 28.70% | 66.80% | 70.5% |
| **Vision Transformer (ViT-B/16 Baseline)** | 99.10% | 58.70% | 40.40% | 56.40% | 59.2% |
| **ViT-B/16 + Heavy Augmentation + Few-Shot** | **98.75%** | **76.80%** | **21.95%** | **75.30%** | **77.8%** |

### Key Takeaways
1. **Domain Gap Quantification:** The baseline ResNet-50 experiences an absolute **60.27% accuracy drop** when transferring from PlantVillage to PlantDoc, confirming that clean lab models cannot be directly deployed in the field.
2. **Closing the Gap:** Combining heavy photometric/geometric augmentations with few-shot target adaptation closes the domain gap by **38.32%**, boosting field diagnostic accuracy from $38.15\%$ to $76.80\%$.
3. **ViT Superiority:** ViT-B/16 possesses substantially greater out-of-distribution robustness than standard CNNs (+20.55% higher zero-shot cross-domain accuracy) owing to global self-attention heads that focus on pathological lesion signatures rather than spurious background textures.
4. **Indian Agricultural Application:** For resource-constrained edge devices (e.g. farmer-facing Android apps on budget smartphones), an adapted **MobileNetV2** provides the optimal balance of edge efficiency (<35ms inference latency) and robust field diagnosis.

---

## 🚀 Repository Structure
```
DL/
├── notebooks/
│   └── DL_Project_P05_Domain_Gap_Colab.ipynb  # Interactive Google Colab Notebook
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Dataset mapping, loaders, and augmentation pipelines
│   ├── models.py         # ResNet-50, MobileNetV2, and ViT-B/16 architectures
│   ├── train.py          # Training loop with validation and checkpoints
│   ├── evaluate.py       # Cross-domain evaluator, domain gap metric, and tables
│   ├── bg_removal.py     # HSV leaf lesion segmentation and background masking
│   └── few_shot_adapt.py # Few-shot target domain adaptation
├── report/
│   ├── DL_Project_P05_Report.pdf # Final Compiled Research Paper (PDF)
│   └── Elcvia2e.tex              # LaTeX Research Report Source
├── report_template/
│   └── Elcvia2e.tex      # LaTeX Research Report (ELCVIA Template)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Quickstart & Execution

### 1. Run on Google Colab (Recommended)
Open `notebooks/DL_Project_P05_Domain_Gap_Colab.ipynb` directly in Google Colab with free T4 GPU. Run all cells sequentially to download datasets, train models, and generate benchmark tables and plots.

### 2. Local Setup
```bash
# Clone the repository
git clone <your-repo-link>
cd DL

# Install dependencies
pip install -r requirements.txt

# Run full cross-domain training & evaluation
python -m src.train
```

---

## 📚 References
1. **Mohanty et al. (2016)**: *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science. [arXiv:1604.03169](https://arxiv.org/abs/1604.03169)
2. **Singh et al. (2020)**: *PlantDoc: A Dataset for Visual Plant Disease Detection in Natural Field Conditions*, ACM CoDS-COMAD. [arXiv:1911.10317](https://arxiv.org/abs/1911.10317)
3. **Dosovitskiy et al. (2020)**: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
