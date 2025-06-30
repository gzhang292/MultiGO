# MultiGO

*MultiGO: Towards Multi-level Geometry Learning for Monocular 3D Textured Human Reconstruction*

## Introduction

MultiGO is a model designed for monocular 3D human reconstruction, aiming to achieve high-quality reconstruction through multi-level geometry learning.

## Model Parameters Download

Before using MultiGO for inference, you need to download the model parameters. Please follow the steps below to obtain the required parameters:

1. Click the following link to download the model parameters:
   - [Download Model Parameters](https://drive.google.com/file/d/1JJDk-oo588froso33BM62FedXVoZBTlN/view)

2. Extract the downloaded file and place the parameter files in the `workspace` directory of the project.

   - multigo/workspace/model.safetensors

## Environment

1. To use MultiGO, you need to set up the appropriate PyTorch environment. Please refer to the following repositories to prepare your environment:

- [LGM Repository](https://github.com/3DTopia/LGM)
- [SiTH Repository](https://github.com/SiTH-Diffusion/SiTH)

Make sure to follow the instructions in these repositories to install the necessary dependencies and set up your environment correctly.

2. Download  [SMPL-X](https://smpl-x.is.tue.mpg.de/) models and move them to the `smpl_estimated_related/data/body_models` folder. You should have the following data structure:
```
body_models
    └──smplx
        ├── SMPLX_NEUTRAL.pkl
        ├── SMPLX_NEUTRAL.npz
        ├── SMPLX_MALE.pkl
        ├── SMPLX_MALE.npz
        ├── SMPLX_FEMALE.pkl
        └── SMPLX_FEMALE.npz
```

## Usage

Once your environment is set up, you can run the inference script by executing the following command:

```bash
bash infer.sh
