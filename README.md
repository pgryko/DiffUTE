Fork of original DiffUTE code, added comments and restructured code for better understanding and readability

# DiffUTE

This repository is the code of our NeurIPS'23 paper "DiffUTE: Universal Text Editing Diffusion Model". 

:tada::tada::tada: We have reproduced our method using the AnyText dataset, and the weights can be downloaded at the [URL](https://modelscope.cn/models/cccnju/DiffUTE_SD2_Inp/summary). This checkpoint can be load for a [demo](https://github.com/chenhaoxing/DiffUTE/blob/main/app.ipynb).
![](docs/ute.png)
## Getting Started with DiffUTE
### Installation
The codebases are built on top of [diffusers](https://github.com/huggingface/diffusers). Thanks very much.

#### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.10.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV
- transformers
#### Steps
1. Install diffusers following https://github.com/huggingface/diffusers.

2. Prepare datasets. Due to data sensitivity issues, our data will not be publicly available now, you can reproduce it on your own data, and all images with text are available for model training. Because our data is present on [Ali-Yun oss](https://www.aliyun.com/search?spm=5176.22772544.J_8058803260.37.4aa92ea9DAomsC&k=OSS&__is_mobile__=false&__is_spider__=false&__is_grey__=false), we have chosen pcache to read the data we have stored. You can change the data reading method according to the way you store the data.

3. Train VAE

4. Train DiffUTE


## Experimental results
![](docs/result.png)

## Citing DiffUTE

If you use DiffUTE in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@inproceedings{DiffUTE,
      title={DiffUTE: Universal Text Editing Diffusion Model},
      author={Chen, Haoxing and Xu, Zhuoer and Gu, Zhangxuan and Lan, Jun and Zheng, Xing and Li, Yaohui and Meng, Changhua and Zhu, Huijia and Wang, Weiqiang},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
      year={2023}
}
```

## Contacts
Please feel free to contact us if you have any problems.

Email: [hx.chen@hotmail.com](hx.chen@hotmail.com) or [zhuoerxu.xzr@antgroup.com](zhuoerxu.xzr@antgroup.com)

# DiffUTE Training Scripts V2

This repository contains updated training scripts for the DiffUTE (Diffusion Universal Text Editor) model. The scripts have been modernized with improved data handling, better code organization, and MinIO integration for efficient data storage.

## Key Changes

1. Replaced pcache_fileio with MinIO for data handling
2. Removed alps dependencies
3. Improved code organization and readability
4. Enhanced error handling and logging
5. Better type hints and documentation
6. Modernized training loops

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── README.md
├── requirements.txt
├── train_vae_v2.py
├── train_diffute_v2.py
└── utils/
    └── minio_utils.py
```

## Training Scripts

### VAE Training

Train the VAE component using:

```bash
python train_vae_v2.py \
    --pretrained_model_name_or_path "path/to/model" \
    --output_dir "vae-fine-tuned" \
    --data_path "path/to/data.csv" \
    --resolution 512 \
    --train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --minio_endpoint "your-minio-endpoint" \
    --minio_access_key "your-access-key" \
    --minio_secret_key "your-secret-key" \
    --minio_bucket "your-bucket-name"
```

### DiffUTE Training

Train the complete DiffUTE model using:

```bash
python train_diffute_v2.py \
    --pretrained_model_name_or_path "path/to/model" \
    --output_dir "diffute-fine-tuned" \
    --data_path "path/to/data.csv" \
    --resolution 512 \
    --train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --guidance_scale 0.8 \
    --minio_endpoint "your-minio-endpoint" \
    --minio_access_key "your-access-key" \
    --minio_secret_key "your-secret-key" \
    --minio_bucket "your-bucket-name"
```

## Data Format

The training data should be specified in a CSV file with the following columns:

For VAE training:
- `path`: Path to the image file in MinIO storage

For DiffUTE training:
- `image_path`: Path to the image file in MinIO storage
- `ocr_path`: Path to the OCR results JSON file in MinIO storage

## MinIO Setup

1. Install and configure MinIO server
2. Create a bucket for storing training data
3. Upload your training images and OCR results
4. Configure access credentials in the training scripts

## Model Architecture

The DiffUTE model consists of three main components:

1. VAE (Variational AutoEncoder):
   - Handles image encoding/decoding
   - Pre-trained and frozen during DiffUTE training
   - Reduces computational complexity by working in latent space

2. UNet:
   - Main trainable component
   - Performs denoising in latent space
   - Conditioned on text embeddings
   - Takes concatenated input of noisy latents, mask, and masked image

3. TrOCR:
   - Pre-trained text recognition model
   - Provides text embeddings for conditioning
   - Frozen during training

## Training Process

1. Data Preparation:
   - Images are loaded from MinIO storage
   - OCR results are used to identify text regions
   - Images are preprocessed and normalized

2. Training Loop:
   - VAE encodes images to latent space
   - Random noise is added according to diffusion schedule
   - UNet predicts noise or velocity
   - Loss is calculated and model is updated
   - Checkpoints are saved periodically

## Error Handling

The scripts include robust error handling:
- Graceful handling of failed image loads
- Fallback mechanisms for missing data
- Detailed logging of errors
- Proper cleanup of resources

## Contributing

All rights go to original authors
