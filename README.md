# Deblur Project Setup

## Requirements
- Python 3.9
- CUDA 11.8

## Environment Setup

1. Create and activate a new conda environment:
```
conda create -n deblur python=3.9
conda activate deblur
```

2. Install required Python packages:
```
pip install -r requirements.txt
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## Model Setup

1. Update the model path in `./models/NAFNet-width64.yml` if needed.

2. Download the models from the following link:
[Download Models](https://drive.usercontent.google.com/download?id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X&authuser=0)
