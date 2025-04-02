# -*- coding: utf-8 -*-
"""
Script for fine-tuning the Variational Autoencoder (VAE) component of a
pre-trained Stable Diffusion model.

This script focuses solely on optimizing the VAE for better image reconstruction,
potentially improving the quality or style adaptation of generated images.
It utilizes the `diffusers` library for model handling, `accelerate` for
distributed training and mixed-precision support, and `Minio` for loading
data from an S3-compatible object store.
"""

# --- Standard Library Imports ---
import argparse
import io
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Third-Party Imports ---
import albumentations as alb
import cv2  # OpenCV for image loading and processing
import datasets
import diffusers
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from albumentations.pytorch import ToTensorV2
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from minio import Minio
from minio.error import S3Error
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Note: CLIPTextModel, CLIPTokenizer are removed as they are not used.

# --- Environment Setup ---

# Ensure a minimum version of diffusers is installed.
check_min_version("0.15.0.dev0")

# Set CUDA environment variable for debugging asynchronous kernel launches (optional).
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set the multiprocessing sharing strategy to 'file_system'
# This can help avoid "Too many open files" errors in certain environments.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    # Handles cases where the strategy might already be set or not applicable
    pass

# Get a logger instance configured by accelerate.
logger = get_logger(__name__, log_level="INFO")


# ==========================================
#          MinIO Configuration
# ==========================================

# Load MinIO connection details from environment variables with defaults.
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "play.min.io")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "your-access-key")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "your-secret-key")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "your-bucket")
MINIO_SECURE = os.getenv("MINIO_SECURE", "True").lower() == "true"


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    """
    Creates and returns a singleton MinIO client instance.

    Uses `lru_cache` to ensure only one client object is created.

    Returns:
        Minio: An initialized MinIO client object.

    Raises:
        ValueError: If the configured bucket does not exist.
        Exception: If the client fails to initialize for other reasons.
    """
    try:
        logger.info(
            f"Initializing MinIO client for endpoint: {MINIO_ENDPOINT}, bucket: {MINIO_BUCKET}"
        )
        client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        # Check if the bucket exists
        found = client.bucket_exists(MINIO_BUCKET)
        if not found:
            logger.error(f"MinIO bucket '{MINIO_BUCKET}' not found or accessible.")
            raise ValueError(f"MinIO bucket '{MINIO_BUCKET}' not found or accessible.")
        logger.info(f"Successfully connected to MinIO bucket '{MINIO_BUCKET}'.")
        return client
    except S3Error as e:
        logger.error(
            f"MinIO S3 error during client initialization or bucket check: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to create MinIO client: {e}")
        raise


def download_image_minio(file_path: str) -> Optional[np.ndarray]:
    """
    Downloads an image file from the configured MinIO bucket and decodes it
    into a NumPy array (OpenCV format BGR).

    Args:
        file_path (str): The path (key) of the file within the MinIO bucket.

    Returns:
        Optional[np.ndarray]: The decoded image as a NumPy array (BGR format),
                              or None if download/decoding fails.
    """
    try:
        client = get_minio_client()
        logger.debug(
            f"Attempting to download from MinIO: bucket='{MINIO_BUCKET}', path='{file_path}'"
        )

        data = client.get_object(MINIO_BUCKET, file_path)
        buffer = io.BytesIO()
        for d in data.stream(32 * 1024):
            buffer.write(d)
        buffer.seek(0)

        file_bytes = np.frombuffer(buffer.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning(f"Failed to decode image from MinIO path: {file_path}")
            return None

        logger.debug(
            f"Successfully downloaded and decoded image from MinIO: {file_path}"
        )
        return img
    except S3Error as e:
        logger.error(f"MinIO S3 error while downloading {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing file {file_path} from MinIO: {e}")
        return None


# ==========================================
#          Dataset Definition
# ==========================================


def create_image_transforms(
    resolution: int, center_crop: bool, random_flip: bool
) -> Tuple[alb.Compose, alb.Compose]:
    """
    Creates Albumentations transformation pipelines for image preprocessing.

    Args:
        resolution (int): Target resolution for the images.
        center_crop (bool): If True, use CenterCrop; otherwise, use RandomCrop.
        random_flip (bool): If True, add HorizontalFlip augmentation.

    Returns:
        Tuple[alb.Compose, alb.Compose]: A tuple containing:
            - transform_resize_crop: Pipeline for resizing, cropping, flipping, and normalization.
            - transform_to_tensor: Pipeline for converting to PyTorch tensor.
    """
    crop_transform = (
        alb.CenterCrop(height=resolution, width=resolution, p=1.0)
        if center_crop
        else alb.RandomCrop(height=resolution, width=resolution, p=1.0)
    )

    resize_crop_list = [
        # Consider adding SmallestMaxSize before crop if input sizes vary significantly below target resolution
        # alb.SmallestMaxSize(max_size=resolution, interpolation=cv2.INTER_AREA, p=1.0),
        crop_transform,
        alb.Resize(
            height=resolution, width=resolution, interpolation=cv2.INTER_AREA
        ),  # Ensure final size
    ]
    if random_flip:
        resize_crop_list.append(alb.HorizontalFlip(p=0.5))

    resize_crop_list.append(
        alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    )  # Normalize to [-1, 1]

    transform_resize_crop = alb.Compose(resize_crop_list)
    transform_to_tensor = alb.Compose(
        [ToTensorV2()]
    )  # Converts HWC NumPy array to CHW PyTorch Tensor

    return transform_resize_crop, transform_to_tensor


class VAEDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images for VAE fine-tuning from MinIO.

    Reads image paths from a specified CSV file and downloads/processes
    images using MinIO and Albumentations. Handles potential errors during
    loading/processing by logging and skipping problematic images.

    Args:
        csv_path (str): Path to the CSV file containing image paths.
                        Expected to have a column named 'path'.
        resolution (int): Target resolution for images.
        center_crop (bool): Whether to use center cropping instead of random cropping.
        random_flip (bool): Whether to apply random horizontal flipping.
    """

    def __init__(
        self,
        csv_path: str,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = False,
    ):
        self.csv_path = csv_path
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.image_paths = self._load_image_paths()
        self.num_images = len(self.image_paths)

        if self.num_images == 0:
            raise ValueError(f"No valid image paths found in {self.csv_path}.")

        self.transform_resize_crop, self.transform_to_tensor = create_image_transforms(
            resolution=self.resolution,
            center_crop=self.center_crop,
            random_flip=self.random_flip,
        )
        logger.info(
            f"Initialized VAEDataset with {self.num_images} images from {self.csv_path}."
        )
        logger.info(
            f"Transforms: resolution={resolution}, center_crop={center_crop}, random_flip={random_flip}"
        )

    def _load_image_paths(self) -> List[str]:
        """Loads image file paths from the specified CSV file."""
        try:
            logger.info(f"Loading training image paths from {self.csv_path}...")
            df = pd.read_csv(self.csv_path, low_memory=False)
            if "path" not in df.columns:
                raise ValueError(
                    f"CSV file '{self.csv_path}' must contain a 'path' column."
                )
            paths = df["path"].dropna().astype(str).tolist()
            logger.info(f"Found {len(paths)} paths in {self.csv_path}.")
            return paths
        except FileNotFoundError:
            logger.error(f"Error: CSV file not found at {self.csv_path}.")
            raise
        except Exception as e:
            logger.error(f"Error loading or processing {self.csv_path}: {e}")
            raise

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return self.num_images

    def __getitem__(self, index: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Gets a single data point (processed image) from the dataset.

        Downloads the image from MinIO, applies preprocessing and augmentations.
        Returns None if the image cannot be loaded or processed.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            Optional[Dict[str, torch.Tensor]]: A dictionary containing the
            processed image tensor under the key "pixel_values", or None if
            an error occurs.
        """
        image_path = self.image_paths[index]
        try:
            # --- Download Image ---
            instance_image = download_image_minio(image_path)
            if instance_image is None:
                logger.warning(
                    f"Skipping image at index {index} (path: {image_path}) due to download/decode error."
                )
                return None  # Signal to collate_fn to skip this item

            # Convert BGR (OpenCV default) to RGB
            instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)

            # --- Pre-resize (Optional but Recommended for small images) ---
            h, w, _ = instance_image.shape
            min_side = min(h, w)
            target_size = self.resolution
            # Upscale if image is smaller than target resolution to avoid low-detail crops
            if min_side < target_size:
                scale_factor = target_size / min_side
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                interpolation = cv2.INTER_LANCZOS4  # Good for enlarging
                instance_image = cv2.resize(
                    instance_image, (new_w, new_h), interpolation=interpolation
                )
                logger.debug(
                    f"Resized image from {(h, w)} to {(new_h, new_w)} before cropping (path: {image_path})."
                )

            # --- Apply Transformations ---
            augmented = self.transform_resize_crop(image=instance_image)
            instance_image_np = augmented["image"]  # Normalized numpy array (H, W, C)

            augmented_tensor = self.transform_to_tensor(image=instance_image_np)
            instance_image_tensor = augmented_tensor["image"]  # CHW PyTorch tensor

            return {"pixel_values": instance_image_tensor}

        except Exception as e:
            logger.error(
                f"Error processing image at index {index} (path: {image_path}): {e}",
                exc_info=True,
            )
            return None  # Signal to collate_fn to skip


def collate_fn(
    batch: List[Optional[Dict[str, torch.Tensor]]],
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function for the DataLoader.

    Filters out None values (errors during __getitem__) and stacks valid examples.
    Returns None if the entire batch is invalid.

    Args:
        batch: A list of dictionaries (or None) from the dataset's __getitem__.

    Returns:
        Optional[Dict[str, torch.Tensor]]: A dictionary containing a batch of
        pixel values under the key "pixel_values", or None if the batch is empty.
    """
    # Filter out None entries caused by errors in __getitem__
    valid_examples = [item for item in batch if item is not None]

    if not valid_examples:
        logger.warning(
            "Collate function received an empty or fully invalid batch. Skipping."
        )
        return None  # Skip this batch entirely

    # Stack the pixel values from valid examples
    pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])

    # Ensure tensor is in contiguous memory format and correct dtype
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"pixel_values": pixel_values}


# ==========================================
#          Argument Parsing
# ==========================================


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Script for fine-tuning the VAE of a Stable Diffusion model."
    )

    # --- Model and Data Paths ---
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained Stable Diffusion model or identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific model version (branch, tag, commit hash) to use.",
    )
    parser.add_argument(
        "--data_csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing image paths (must have a 'path' column for MinIO keys).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-vae-finetuned",
        help="Directory where model checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching downloaded models.",
    )

    # --- Training Hyperparameters ---
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Input image resolution. Images will be resized/cropped.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Use center cropping instead of random cropping.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        default=True,  # Defaulting to True as common practice
        help="Apply random horizontal flipping.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size per GPU/TPU device."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=50,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. Overrides num_train_epochs if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before optimizer update.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing in VAE to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale learning rate based on GPUs, batch size, and accumulation steps.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for the learning rate scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer (requires bitsandbytes).",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 precision on Ampere GPUs.",
    )
    # --use_ema argument removed as EMA is typically applied to the UNet, not the VAE in standard SD finetuning.
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam optimizer beta1."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Adam optimizer beta2."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08, help="Adam optimizer epsilon."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Maximum gradient norm for clipping.",
    )

    # --- Logging, Checkpointing, and Hub ---
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the final model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hugging Face Hub authentication token.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Repository name on the Hub (e.g., your-username/sd-vae-finetuned).",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs (relative to output_dir).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training ('no', 'fp16', 'bf16'). Defaults to accelerator config.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Log reporting integration(s). Options: "tensorboard", "wandb", "comet_ml", "all".',
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Internal use for distributed training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a training checkpoint every X steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,  # Keep all checkpoints by default
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume training from a checkpoint directory or "latest".',
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Use xFormers in UNet (if available). UNet is frozen but might be used for eval.",
    )
    # --non_ema_revision argument removed as EMA is not used for VAE.

    args = parser.parse_args()

    # --- Post-processing and Sanity Checks ---
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        logger.info(
            f"Overriding local_rank {args.local_rank} with environment variable LOCAL_RANK {env_local_rank}"
        )
        args.local_rank = env_local_rank

    if not os.path.exists(args.data_csv_path):
        raise FileNotFoundError(f"Data CSV file not found at: {args.data_csv_path}")

    # Ensure output directory exists if not pushing to hub
    if not args.push_to_hub and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


# ==========================================
#          Hugging Face Hub Utils
# ==========================================


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
) -> str:
    """Constructs the full repository name for the Hugging Face Hub."""
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        try:
            username = whoami(token)["name"]
            return f"{username}/{model_id}"
        except Exception as e:
            logger.error(
                f"Could not get username from Hub token: {e}. Ensure login or provide organization."
            )
            raise ValueError("Could not determine Hub username.") from e
    else:
        return f"{organization}/{model_id}"


# ==========================================
#               Main Function
# ==========================================


def main():
    """Main function to orchestrate the VAE fine-tuning process."""
    args = parse_args()

    # --- 1. Setup Environment (Accelerator, Logging, Seed) ---
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. Handle Hub Repository ---
    repo = None  # Initialize repo to None
    if accelerator.is_main_process and args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(args.output_dir).name, token=args.hub_token
            )
        else:
            repo_name = args.hub_model_id
        logger.info(f"Creating or using Hub repository: {repo_name}")
        create_repo(repo_name, exist_ok=True, token=args.hub_token)
        repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)
        # Add patterns to .gitignore
        with open(os.path.join(args.output_dir, ".gitignore"), "a+") as gitignore:
            gitignore.seek(0)
            content = gitignore.read()
            if "step_*" not in content:
                gitignore.write("step_*\n")
            if "epoch_*" not in content:
                gitignore.write("epoch_*\n")

    # --- 3. Load Models (VAE and UNet) ---
    logger.info(f"Loading VAE from: {args.pretrained_model_name_or_path}")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    logger.info(
        f"Loading UNet from: {args.pretrained_model_name_or_path} (will be frozen)"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        cache_dir=args.cache_dir,
        # Using args.revision for UNet as non_ema_revision is deprecated/removed
    )

    # --- 4. Configure Models (Freeze UNet, xFormers, Grad Checkpointing, TF32) ---
    unet.requires_grad_(False)
    logger.info("UNet parameters frozen.")
    # VAE parameters remain trainable by default.

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xFormers memory efficient attention for UNet.")
            except Exception as e:
                logger.warning(
                    f"Could not enable xFormers: {e}. Proceeding without it."
                )
        else:
            logger.warning("xFormers is not available.")

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing for VAE.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("Enabled TF32 precision for matmul.")

    # --- 5. Prepare Optimizer ---
    if args.scale_lr:
        scaled_lr = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        logger.info(f"Scaling learning rate from {args.learning_rate} to {scaled_lr}")
        args.learning_rate = scaled_lr

    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            logger.warning(
                "bitsandbytes not found, falling back to standard AdamW. Install with `pip install bitsandbytes`"
            )

    optimizer = optimizer_cls(
        vae.parameters(),  # Optimize only VAE parameters
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- 6. Prepare Dataset and DataLoader ---
    logger.info("Initializing dataset...")
    train_dataset = VAEDataset(
        csv_path=args.data_csv_path,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )

    logger.info("Initializing DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,  # Use the custom collate function
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for simplicity
    )

    # --- 7. Prepare LR Scheduler ---
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(
            f"Calculated max_train_steps: {args.max_train_steps} ({args.num_train_epochs} epochs)"
        )
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
        overrode_max_train_steps = False
        logger.info(
            f"Using provided max_train_steps: {args.max_train_steps}. Adjusted num_train_epochs: {args.num_train_epochs}"
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    logger.info(
        f"Using LR scheduler: {args.lr_scheduler} with {args.lr_warmup_steps} warmup steps."
    )

    # --- 8. Prepare with Accelerator ---
    logger.info(
        "Preparing VAE, optimizer, dataloader, and scheduler with accelerate..."
    )
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )
    # Move frozen UNet to the correct device manually
    unet = unet.to(accelerator.device)

    # Define weight dtype based on mixed precision setting
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to the correct dtype (VAE handled by accelerator, UNet manually)
    unet.to(dtype=weight_dtype)
    logger.info(f"Using weight dtype: {weight_dtype}")

    # Recalculate steps/epochs after dataloader preparation by accelerator
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # --- 9. Initialize Trackers ---
    if accelerator.is_main_process:
        run_name = f"vae_finetune_{Path(args.pretrained_model_name_or_path).name}_lr{args.learning_rate}_bs{args.train_batch_size}"
        accelerator.init_trackers(run_name, config=vars(args))
        logger.info(
            f"Initialized trackers ({args.report_to}) with run name: {run_name}"
        )

    # --- 10. Training Loop Setup ---
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Starting VAE Fine-tuning *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, dist & accum) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    # --- 11. Resume from Checkpoint ---
    if args.resume_from_checkpoint:
        resume_path = None
        if args.resume_from_checkpoint != "latest":
            resume_path = args.resume_from_checkpoint
        else:
            # Find the most recent checkpoint
            dirs = [
                d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")
            ]
            if dirs:
                dirs.sort(key=lambda x: int(x.split("-")[1]))
                resume_path = os.path.join(args.output_dir, dirs[-1])

        if resume_path and os.path.isdir(resume_path):
            try:
                logger.info(f"Resuming training from checkpoint: {resume_path}")
                accelerator.load_state(resume_path)
                global_step = int(resume_path.split("-")[-1])
                logger.info(f"Resumed from global step: {global_step}")
                # Calculate start epoch and step within epoch
                resume_global_step = global_step * args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * args.gradient_accumulation_steps
                )
                logger.info(
                    f"Resuming at Epoch {first_epoch + 1}, Step {resume_step} (in dataloader)."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load checkpoint from {resume_path}: {e}. Starting from scratch."
                )
                args.resume_from_checkpoint = None  # Reset flag
                global_step = 0
                first_epoch = 0
                resume_step = 0
        else:
            logger.warning(
                f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting from scratch."
            )
            args.resume_from_checkpoint = None

    # --- 12. Initialize Progress Bar ---
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="VAE Training Steps",
    )

    # ==============================
    #       >>> Training Loop <<<
    # ==============================
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
        vae.train()  # Set VAE to training mode
        train_loss_accum = 0.0  # Accumulate loss over logging steps

        for step, batch in enumerate(train_dataloader):

            # Skip steps if resuming within an epoch
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(
                        1
                    )  # Still update progress bar for skipped optimizer steps
                continue

            # Handle cases where collate_fn returns None (empty/invalid batch)
            if batch is None:
                logger.warning(
                    f"Skipping step {step} in epoch {epoch+1} due to invalid batch."
                )
                continue

            # Perform training step under accelerator's context manager
            with accelerator.accumulate(vae):
                # --- Forward Pass ---
                input_img = batch["pixel_values"].to(
                    dtype=weight_dtype
                )  # Already on correct device via prepare

                # Pass input images through the VAE
                # VAE output is DiagonalGaussianDistribution, access sample via `.sample`
                reconstruction = vae(input_img).sample

                # --- Calculate Loss ---
                # MSE loss between original input and reconstruction
                # Cast to float32 for stable loss calculation
                loss = F.mse_loss(
                    reconstruction.float(), input_img.float(), reduction="mean"
                )

                # --- Logging Loss ---
                # Gather loss across all devices for accurate average logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss_accum += avg_loss.item() / args.gradient_accumulation_steps

                # --- Backward Pass & Optimization ---
                accelerator.backward(loss)

                if (
                    accelerator.sync_gradients
                ):  # Only clip when gradients are synchronized
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )  # Use set_to_none=True for potential memory savings

            # --- Post-Optimization Step ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                accelerator.log({"train_loss": train_loss_accum}, step=global_step)
                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                train_loss_accum = 0.0  # Reset accumulated loss

                # --- Checkpointing ---
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                        # Optional: Push checkpoint to Hub
                        if args.push_to_hub and repo:
                            try:
                                logger.info(
                                    f"Pushing checkpoint {global_step} to Hub..."
                                )
                                repo.push_to_hub(
                                    commit_message=f"Training checkpoint step {global_step}",
                                    blocking=False,
                                )
                            except Exception as e:
                                logger.error(f"Failed to push checkpoint to Hub: {e}")

            # Check if max training steps reached
            if global_step >= args.max_train_steps:
                logger.info("Maximum training steps reached.")
                break  # Exit inner loop

        # Check again after epoch end
        if global_step >= args.max_train_steps:
            break  # Exit outer loop

    # --- 13. End of Training ---
    logger.info("Training finished.")
    accelerator.wait_for_everyone()

    # --- 14. Save Final Model ---
    if accelerator.is_main_process:
        vae_final = accelerator.unwrap_model(vae)
        # Save the fine-tuned VAE weights
        final_save_path = os.path.join(args.output_dir, "final_vae")
        vae_final.save_pretrained(final_save_path)
        logger.info(f"Saved final fine-tuned VAE model to {final_save_path}")

        # --- Push Final Model to Hub ---
        if args.push_to_hub and repo:
            try:
                logger.info(f"Pushing final model to Hub repository: {repo.repo_id}")
                # Upload the entire output directory content
                repo.push_to_hub(commit_message="End of VAE training", blocking=True)
                logger.info("Successfully pushed final model to Hub.")
            except Exception as e:
                logger.error(f"Failed to push final model to Hub: {e}")

    # --- 15. Clean Up ---
    accelerator.end_training()
    logger.info("Accelerator training ended.")


if __name__ == "__main__":
    main()
