"""
Script for fine-tuning the Variational Autoencoder (VAE) component of a pre-trained Stable Diffusion model.

This script focuses on optimizing the VAE for better image reconstruction,
using MinIO for efficient data loading and distributed training capabilities.
"""

import os
import logging
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
import albumentations as alb
from albumentations.pytorch import ToTensorV2

from utils.minio_utils import MinioHandler

# Will error if the minimal version of diffusers is not installed
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parse training arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="VAE fine-tuning script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vae-fine-tuned",
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


class TrainingDataset(Dataset):
    """Dataset for VAE training using MinIO storage."""

    def __init__(
        self,
        minio_handler: MinioHandler,
        data_paths: list,
        resolution: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            minio_handler: Initialized MinIO handler
            data_paths: List of image paths in MinIO
            resolution: Target image resolution
        """
        self.minio = minio_handler
        self.image_paths = data_paths
        self.resolution = resolution

        self.transform = alb.Compose(
            [
                alb.SmallestMaxSize(max_size=resolution),
                alb.CenterCrop(resolution, resolution),
                alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            # Load and preprocess image
            image = self.minio.download_image(image_path)
            transformed = self.transform(image=image)
            return {"pixel_values": transformed["image"]}
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a random noise image as fallback
            return {"pixel_values": torch.randn(3, self.resolution, self.resolution)}


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
    image_paths = df["path"].tolist()

    # Create dataset and dataloader
    dataset = TrainingDataset(
        minio_handler=minio_handler,
        data_paths=image_paths,
        resolution=args.resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Load VAE model
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(),
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
    vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, dataloader, lr_scheduler
    )

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
        vae.train()
        train_loss = 0.0

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(vae):
                # Get VAE loss
                loss = vae(batch["pixel_values"], return_dict=False)[0]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), 1.0)

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
            pipeline = accelerator.unwrap_model(vae)
            pipeline.save_pretrained(
                os.path.join(args.output_dir, f"checkpoint-{global_step}")
            )

        train_loss = train_loss / num_update_steps_per_epoch
        logger.info(f"Epoch {epoch}: Average loss = {train_loss:.4f}")


if __name__ == "__main__":
    main()
