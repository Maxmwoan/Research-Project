# "How Much Data is Enough?" Modelling Learning Curves

This repo is used for student research project CSE3000 only. 

Acknowledgement: The code under the [`lcpfn`](./lcpfn/) folder is originally from the [LC-PFN](https://github.com/automl/lcpfn) project.


### Environment Setup Guide
To run the code, we recommend creating a clean virtual environment using [conda](https://repo.anaconda.com/miniconda/) and installing the dependencies step by step. 

1. Create a virtual environment 
```bash
conda create -n lc-env python=3.12
conda activate lc-env
```
2. Install Required Dependencies
```bash
pip install -r requirements.txt
```
3. Install PyTorch (based on your devices, see [PyTorch](https://pytorch.org/get-started/locally/)). 
Example (for CUDA 12.6, GPU-enabled):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```