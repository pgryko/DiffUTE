"""
DiffUTE (Diffusion Universal Text Editor) Training Script V2

This script implements the training process for the DiffUTE model, which edits text
in images while preserving surrounding context. The implementation uses MinIO for
efficient data loading and modern PyTorch practices for training.

Key Components:
- VAE: Pre-trained and frozen, handles image encoding/decoding
- UNet: Main trainable component for denoising in latent space
- TrOCR: Pre-trained text recognition for conditioning
"""

import os
import logging
import math
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.minio_utils import MinioHandler

# Will error if the minimal version of diffusers is not installed
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parse training arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="DiffUTE training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffute-model-finetuned",
        help="Output directory for checkpoints and models",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CSV file containing training data paths",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for training images",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.8,
        help="Scale for guidance loss",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    # MinIO configuration
    parser.add_argument("--minio_endpoint", type=str, required=True)
    parser.add_argument("--minio_access_key", type=str, required=True)
    parser.add_argument("--minio_secret_key", type=str, required=True)
    parser.add_argument("--minio_bucket", type=str, required=True)

    args = parser.parse_args()
    return args


def prepare_mask_and_masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a masked version of the input image.

    Args:
        image: Input image array (H, W, C)
        mask: Binary mask array (H, W)

    Returns:
        Masked image with target regions set to 0
    """
    return image * np.stack([mask < 0.5] * 3, axis=2)


def generate_mask(size: Tuple[int, int], bbox: list, dilation: int = 10) -> np.ndarray:
    """
    Generate a binary mask for the text region.

    Args:
        size: (width, height) of the mask
        bbox: [x1, y1, x2, y2] text bounding box
        dilation: Number of pixels to dilate the mask

    Returns:
        Binary mask array
    """
    mask = np.zeros(size[::-1], dtype=np.float32)
    x1, y1, x2, y2 = map(int, bbox)
    mask[y1:y2, x1:x2] = 1

    if dilation > 0:
        import cv2

        kernel = np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


class DiffUTEDataset(Dataset):
    """Dataset for DiffUTE training using MinIO storage."""

    def __init__(
        self,
        minio_handler: MinioHandler,
        data_paths: list,
        ocr_paths: list,
        resolution: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            minio_handler: Initialized MinIO handler
            data_paths: List of image paths in MinIO
            ocr_paths: List of OCR result paths in MinIO
            resolution: Target image resolution
        """
        self.minio = minio_handler
        self.image_paths = data_paths
        self.ocr_paths = ocr_paths
        self.resolution = resolution

        self.transform = alb.Compose(
            [
                alb.SmallestMaxSize(max_size=resolution),
                alb.CenterCrop(resolution, resolution),
                alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )

        self.mask_transform = alb.Compose(
            [
                alb.SmallestMaxSize(max_size=resolution),
                alb.CenterCrop(resolution, resolution),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        ocr_path = self.ocr_paths[idx]

        try:
            # Load image and OCR results
            image = self.minio.download_image(image_path)
            ocr_data = self.minio.read_json(ocr_path)

            # Process OCR results
            ocr_df = pd.DataFrame(ocr_data["document"])
            ocr_df = ocr_df[ocr_df["score"] > 0.8]

            if len(ocr_df) == 0:
                raise ValueError("No valid OCR results found")

            # Randomly select one text region
            ocr_sample = ocr_df.sample(n=1).iloc[0]
            text = ocr_sample["text"]
            bbox = ocr_sample["box"]

            # Convert bbox to [x1, y1, x2, y2] format
            bbox = [
                min(x[0] for x in bbox),
                min(x[1] for x in bbox),
                max(x[0] for x in bbox),
                max(x[1] for x in bbox),
            ]

            # Generate mask and masked image
            mask = generate_mask(image.shape[:2][::-1], bbox)
            masked_image = prepare_mask_and_masked_image(image, mask)

            # Apply transforms
            transformed = self.transform(image=image)
            transformed_mask = self.mask_transform(image=mask)
            transformed_masked = self.transform(image=masked_image)

            return {
                "pixel_values": transformed["image"],
                "mask": transformed_mask["image"][0],
                "masked_image": transformed_masked["image"],
                "text": text,
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            # Return random tensors as fallback
            return {
                "pixel_values": torch.randn(3, self.resolution, self.resolution),
                "mask": torch.zeros(self.resolution, self.resolution),
                "masked_image": torch.randn(3, self.resolution, self.resolution),
                "text": "",
            }


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Initialize MinIO handler
    minio_handler = MinioHandler(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket_name=args.minio_bucket,
    )

    # Load data paths
    df = pd.read_csv(args.data_path)
    image_paths = df["image_path"].tolist()
    ocr_paths = df["ocr_path"].tolist()

    # Create dataset and dataloader
    dataset = DiffUTEDataset(
        minio_handler=minio_handler,
        data_paths=image_paths,
        ocr_paths=ocr_paths,
        resolution=args.resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-printed"
    ).encoder

    # Freeze VAE and TrOCR
    vae.requires_grad_(False)
    trocr_model.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )

    # Get number of training steps
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    num_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps,
    )

    # Prepare everything with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # Move models to device and cast to dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    trocr_model.to(accelerator.device, dtype=weight_dtype)

    # Get VAE scale factor
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_train_steps}")

    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Prepare mask
                mask = F.interpolate(
                    batch["mask"].unsqueeze(1),
                    size=latents.shape[2:],
                    mode="nearest",
                )
                mask = mask.to(weight_dtype)

                # Get masked image latents
                masked_image_latents = vae.encode(
                    batch["masked_image"].to(weight_dtype)
                ).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                # Get text embeddings
                text_embeddings = trocr_model(
                    batch["pixel_values"].to(weight_dtype)
                ).last_hidden_state

                # Prepare model input
                model_input = torch.cat(
                    [noisy_latents, mask, masked_image_latents],
                    dim=1,
                )

                # Get model prediction
                model_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                ).sample

                # Calculate loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Scale loss by guidance
                loss = loss * args.guidance_scale

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            train_loss += loss.detach().item()

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Save checkpoint
        if accelerator.is_main_process:
            pipeline = accelerator.unwrap_model(unet)
            pipeline.save_pretrained(
                os.path.join(args.output_dir, f"checkpoint-{global_step}")
            )

        train_loss = train_loss / num_update_steps_per_epoch
        logger.info(f"Epoch {epoch}: Average loss = {train_loss:.4f}")


if __name__ == "__main__":
    main()
