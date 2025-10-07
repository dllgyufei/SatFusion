# SatFusion: A Unified Framework for Enhancing Satellite IoT Images via Multi-Temporal and Multi-Source Data Fusion
Welcome to the official repository of SatFusion,an efficient multimodal fusion framework for IoT-Oriented satellite image enhancement. This repository provides the implementation of Satfusion.
## Introduction
SatFusio is the first to achieve unified end-to-end fusion of multi-temporal and multi-source images by integrating a Multi-Temporal Image Super-Resolution network and a Pansharpening network. The framework consists of three key components: a Multi-Temporal Image Fusion (MTIF) module, a Multi-Source Image Fusion (MSIF) module, and a Fusion Image Composition module. Extensive experiments demonstrate that
SatFusion achieves state-of-the-art performance on both multi-temporal and multi-source remote sensing image fusion tasks while significantly reducing data volume.It also maintains stable performance under challenging conditions such as image blurring and large modality gaps,highlighting its practical potential and scalability.  
![](https://cdn.luogu.com.cn/upload/image_hosting/jt7aqodc.png)  
## Quick Start
### Code Structure
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
The files in forders *src/misr* and *src/sharp* are sub-modules of MISR and Pan-Sharpenning respectively.The file *src/modules.py* is our backbone code.Follow the guidance below for better use of SatFusion.
### Enviroments
CUDA 11.8+  
Python 3.10+  
PyTorch 2.4.0+  
Install additional dependencies by running: 
```
pip install -r requirements.txt  
```
To run the block of Pan-Mamba , Vision-Mamba is required.You can refer to the guidance in [Pan-Mamba](https://github.com/alexhe101/pan-mamba) and this blog [Install Vision Mamba on Linux](https://zhuanlan.zhihu.com/p/687359086).
### Dataset
The dataset we used is Worldstrat.Fetch the entire dataset on [https://worldstrat.github.io/](https://worldstrat.github.io/).
### Train
Set the params *root* as your root dir of the dataset and *list_of_aios* as "pretrained_model/final_split.csv" in file *Train.py*.Run *Train.py* to train the model;    
The process of training is visible on [Weights & Biases](wandb.ai).Replace the *project* and *entity* in *src/train.py*.For details, refer to [Weights & Biases quickstart guide](https://wandb.ai/quickstart?).
### Inference
Set *list_of_aios* as "pretrained_model/predict_split.csv" or replace it with the aios you want.Load the results of training from folder *checkpoints* and set *checkpoint_path* in *Inference.py* as its path.Ensure all other parameters remain consistent with the training configuration.We provide a trained checkpoint of TRNet with INNformer for you to test.
## Issues and Contributions
If you encounter any issues or have suggestions for improvement, please feel free to open an issue in the GitHub issue tracker.   
  
We appreciate your use of SatFusion for your satellite image enhancement needs!We hope it proves to be a valuable framework.
