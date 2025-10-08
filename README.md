# 🌍 SatFusion: A Unified Framework for Enhancing Satellite IoT Images via Multi-Temporal and Multi-Source Data Fusion

> Official Implementation of the WWW 2026 Paper(under review)  
> **SatFusion: A Unified Framework for Enhancing Satellite IoT Images via Multi-Temporal and Multi-Source Data Fusion**

---

## 🛰️ Overview

**SatFusion** is a unified deep learning framework designed to enhance satellite imagery in **Satellite Internet of Things (Sat-IoT)** scenarios by jointly fusing **multi-temporal** and **multi-source** data.

Unlike existing approaches that treat **Multi-Image Super-Resolution (MISR)** and **Pansharpening** as separate tasks, SatFusion integrates both into a single end-to-end architecture to achieve **higher reconstruction quality, robustness, and generalizability**.

![](https://github.com/dllgyufei/SatFusion/blob/main/F3.png)  

---

## 🔬 Key Features

- **Unified Fusion Framework** — Jointly optimizes multi-temporal and multi-source imagery within a single architecture.  
- **Multi-Temporal Image Fusion (MTIF)** — Aggregates temporal complementary information from multiple LRMS inputs.  
- **Multi-Source Image Fusion (MSIF)** — Injects fine-grained spatial textures from high-resolution panchromatic images.  
- **Fusion Composition Module** — Dynamically combines features from both branches with adaptive spectral refinement.  
- **Robustness Under Real-World Conditions** — Performs well under noise, blur, and imperfect alignment common in Sat-IoT imagery.  

---

## 🧩 Architecture

SatFusion is composed of three core modules:

### 1. Multi-Temporal Image Fusion (MTIF)
- **Inputs:** multiple LRMS images 
- **Outputs:** temporally enhanced MS features aligned with the PAN image  
- **Implementation:** shared encoders, feature fusion, and PixelShuffle-based decoding.

### 2. Multi-Source Image Fusion (MSIF)
- **Inputs:** MTIF output + HR PAN image  
- **Outputs:** spatially enhanced MS feature map  
- **Implementation:** builds upon pansharpening modules (e.g., PNN, PanNet, INNformer, Pan-Mamba).

### 3. Fusion Composition Module
- Fuses MTIF and MSIF outputs with residual connections and 1×1 convolutions for adaptive spectral balancing.
---

## ⚙️ Quick Start

### 📁 Code Structure
```
SatFusion/
├── Train_Val_Test.py
├── train.py
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── datasources.py
    ├── lightning_modules.py
    ├── loss.py
    ├── modules.py
    ├── plot.py
    ├── train.py
    ├── transforms.py
    ├── misr/
    │   ├── highresnet.py
    │   ├── misr_public_modules.py
    │   ├── rams.py
    │   ├── srcnn.py
    │   └── trnet.py
    └── sharp/
        ├── mamba/
        |   ├── mamba_module.py
        |   ├── panmamba_baseline_finalversion.py
        |   └── refine_mamba.py      
        ├── pannet/
        |   └── pannet.py
        ├── pnn/
        |   ├── DoubleConv2d.py
        |   └── pnn.py
        └── psit/
            ├── GPPNN_PSIT.py
            ├── modules_psit.py
            └── refine_psit.py
```
> The folders `src/misr` and `src/sharp` contain sub-modules for MISR and Pan-Sharpening, respectively.  
> The file `src/modules.py` serves as the backbone of **SatFusion**.  
---

### 🧱 Environment Setup

- **CUDA:** 11.8+  
- **Python:** 3.10+  
- **PyTorch:** 2.4.0+  

Install additional dependencies:
```bash
pip install -r requirements.txt
```
To run the block of Pan-Mamba , Vision-Mamba is required.You can refer to the guidance in [Pan-Mamba](https://github.com/alexhe101/pan-mamba) and this blog [Install Vision Mamba on Linux](https://zhuanlan.zhihu.com/p/687359086).


### 🌍 Dataset
We conduct experiments on the WorldStrat, WV3, QB, and GF2 datasets.
The complete WorldStrat dataset can be downloaded from [https://worldstrat.github.io/](https://worldstrat.github.io/), while the WV3, QB, and GF2 datasets are available at [https://liangjiandeng.github.io/PanCollection.html](https://liangjiandeng.github.io/PanCollection.html).

This repository provides the full experimental code for the WorldStrat dataset. Experiments on WV3, QB, and GF2 datasets are implemented based on the DLPan-Toolbox framework [https://github.com/liangjiandeng/DLPan-Toolbox.git](https://github.com/liangjiandeng/DLPan-Toolbox.git).


### 🚀 Training, Validation & Testing
Set the params *root* as your root dir of the dataset and *list_of_aios* as "pretrained_model/final_split.csv" in file *Train_Val_Test.py*.Run *Train_Val_Test.py* to train, validate, and test the model.    
The process of training is visible on [Weights & Biases](wandb.ai).Replace the *project* and *entity* in *src/train.py*.For details, refer to [Weights & Biases quickstart guide](https://wandb.ai/quickstart?).



### 🤝 Issues and Contributions
If you encounter any issues or have suggestions for improvement, please feel free to open an issue in the GitHub issue tracker.   
We appreciate your use of SatFusion for your satellite image enhancement needs!We hope it proves to be a valuable framework.
