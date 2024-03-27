## MEATRD: Multimodal Anomalous Tissue Region Detection Enhanced with Spatial Transcriptomics

### Dataset
The raw AnnData files of 10x-hNB can be downloaded from [here](https://cellxgene.cziscience.com/collections/4195ab4c-20bd-4cd3-8b3d-65601277e731)
The raw AnnData files of 10x-hBC-{A1-H1} can be downloaded from [here](https://github.com/almaan/her2st)
The raw AnnData files of 10x-hBC-I1 can be downloaded from [here](https://zenodo.org/records/10437391)

### Code organization
- `configs/` - Configuration files for each task and dataset
- `data/` - The dataset is pre-processed by DGL
- `utils.py` - Util functions (including evaluation suite)
- `main.py` - Main python script for running anomaly detection task
- `pretrain.py` - Python script for pretraining MobileUNet
- `build_datasets.py` - Python scripts for preprocessing
- `MobileUNet.pth` - The pre-trained weights for MobileUNet

In the folder `model`:
- `model/fusion.py` - Multi-modal fusion modules
- `model/meatrd.py` - MEATRD framework for step II and step III
- `model/loss.py` - Loss function of MEATRD
- `model/unet.py` - MobileUNet of MEATRD
