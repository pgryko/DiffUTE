# -*- coding: utf-8 -*-
"""
Fine-tunes a Stable Diffusion UNet model for document image inpainting,
conditioned on text rendered from OCR results.

This script handles:
- Loading document images and OCR data (from local, MinIO, or pcache).
- Preprocessing images, generating masks based on OCR boxes, and rendering text.
- Setting up Stable Diffusion components (VAE, UNet, Scheduler).
- Using a TrOCR model to encode rendered text for conditioning.
- Training the UNet using Accelerate for distributed training and mixed precision.
- Checkpointing and optional pushing to Hugging Face Hub.
"""

# === Standard Library Imports ===
import argparse
import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Machine Learning & Deep Learning (Core) ---
import accelerate

# === Third-Party Library Imports ===
# --- Data Handling & Processing ---
import albumentations as alb
import cv2
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
from datasets import utils as datasets_logging
from diffusers import StableDiffusionPipeline  # Added for saving pipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# --- Cloud Storage & File IO ---
from minio import Minio
from minio.error import S3Error
from packaging import version
from PIL import Image, ImageDraw, ImageFile, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Attempt to import pcache_fileio, handle if not available
try:
    from pcache_fileio import fileio

    PCACHE_AVAILABLE = True
except ImportError:
    PCACHE_AVAILABLE = False
    fileio = None
    print("Warning: pcache_fileio not found. OSS cache functionality disabled.")


# === Configuration & Constants ===
# Ensure minimum diffusers version
check_min_version("0.15.0.dev0")  # Adjust as needed

# Setup Logging
logger = get_logger(__name__, log_level="INFO")  # Accelerate logger

# PIL Configuration
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Multiprocessing Strategy (best effort)
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    logger.warning(
        "Could not set multiprocessing sharing strategy 'file_system'. This might be okay depending on the OS."
    )

# --- Environment Variable Based Configuration ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
MINIO_SECURE = os.getenv("MINIO_SECURE", "True").lower() == "true"
OSS_PCACHE_ROOT_DIR = os.getenv("OSS_PCACHE_ROOT_DIR")

# --- Defaults ---
DEFAULT_FONT_PATH = "arialuni.ttf"  # Ensure this font is accessible
DEFAULT_TROCR_MODEL = "microsoft/trocr-large-printed"
DEFAULT_RESOLUTION = 512
DEFAULT_CROP_SCALE = 256  # Size of the random crop around the text region
DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.8


# === Argument Parsing ===
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion for Document Inpainting with Text Conditioning."
    )

    # --- Paths and Data ---
    data_group = parser.add_argument_group("Data and Paths")
    data_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained Stable Diffusion model or model identifier from Hugging Face Hub.",
    )
    data_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from Hugging Face Hub.",
    )
    data_group.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        help="Revision of non-EMA model identifier (if applicable). Deprecated: Use --variant=non_ema.",
    )
    data_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache pretrained models.",
    )
    data_group.add_argument(
        "--train_data_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing image paths (relative to data source). Must contain a 'path' column.",
    )
    data_group.add_argument(
        "--ocr_data_root",
        type=str,
        required=True,
        help="Root directory/prefix for OCR JSON files (relative to data source).",
    )
    data_group.add_argument(
        "--data_source",
        type=str,
        default="pcache",
        choices=["local", "minio", "pcache"],
        help="Source for loading images and OCR data.",
    )
    data_group.add_argument(
        "--font_path",
        type=str,
        default=DEFAULT_FONT_PATH,
        help="Path to the TTF font file for rendering text.",
    )
    data_group.add_argument(
        "--output_dir",
        type=str,
        default="diffute-model-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # --- Model Configuration ---
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="The resolution for input images and masks.",
    )
    model_group.add_argument(
        "--crop_scale",
        type=int,
        default=DEFAULT_CROP_SCALE,
        help="The size of the square crop region around the text box.",
    )
    model_group.add_argument(
        "--trocr_model_name",
        type=str,
        default=DEFAULT_TROCR_MODEL,
        help="TrOCR model identifier for text feature extraction.",
    )
    model_group.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing.",
    )  # Note: Current dataset doesn't use center_crop explicitly
    model_group.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally.",
    )  # Note: Current dataset doesn't use random_flip

    # --- Training Parameters ---
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Total number of training epochs to perform.",
    )
    train_group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    train_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory.",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    train_group.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    train_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    train_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    train_group.add_argument(
        "--use_8bit_adam", action="store_true", help="Use 8-bit AdamW optimizer."
    )
    train_group.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    train_group.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    train_group.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    train_group.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    train_group.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    train_group.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average for model weights.",
    )
    train_group.add_argument(
        "--ocr_confidence_threshold",
        type=float,
        default=DEFAULT_OCR_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score to consider an OCR detection.",
    )

    # --- Performance & Hardware ---
    perf_group = parser.add_argument_group("Performance and Hardware")
    perf_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    perf_group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU.",
    )
    perf_group.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory-efficient attention.",
    )
    perf_group.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere GPUs (may decrease precision slightly for speed).",
    )
    perf_group.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    perf_group.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    # --- Logging, Checkpointing & Hub ---
    log_group = parser.add_argument_group("Logging, Checkpointing and Hub")
    log_group.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory (subdirectory under output_dir).",
    )
    log_group.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report results to (e.g., "tensorboard", "wandb").',
    )
    log_group.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates steps.",
    )
    log_group.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store. Old ones are deleted.",
    )
    log_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether to resume training from latest checkpoint ('latest') or a specific checkpoint folder name.",
    )
    log_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub.",
    )
    log_group.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    log_group.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository on the Hub.",
    )
    log_group.add_argument(
        "--hub_organization",
        type=str,
        default=None,
        help="The organization name on the Hub (optional).",
    )

    args = parser.parse_args()

    # --- Post-processing and Sanity Checks ---
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank  # Sync local_rank from environment if set

    if not os.path.exists(args.train_data_csv):
        raise ValueError(
            f"Specified --train_data_csv '{args.train_data_csv}' does not exist."
        )
    if not os.path.exists(args.font_path):
        logger.warning(
            f"Specified --font_path '{args.font_path}' does not exist. Text rendering might fail."
        )

    # Check data source requirements
    if args.data_source == "minio" and not all(
        [MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET]
    ):
        raise ValueError(
            "MinIO data source selected, but required environment variables (MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET) are not fully set."
        )
    if args.data_source == "pcache" and not PCACHE_AVAILABLE:
        raise ValueError(
            "pcache data source selected, but pcache_fileio library is not installed."
        )
    if args.data_source == "pcache" and not OSS_PCACHE_ROOT_DIR:
        raise ValueError(
            "pcache data source selected, but OSS_PCACHE_ROOT_DIR environment variable is not set."
        )

    # Default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    # Ensure output directory exists
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        raise ValueError("output_dir must be specified.")

    return args


# === Cloud Storage Access Functions ===


@lru_cache(maxsize=1)
def get_minio_client() -> Optional[Minio]:
    """Creates and returns a cached MinIO client instance."""
    if not all([MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET]):
        logger.warning(
            "MinIO environment variables not fully set. MinIO client not created."
        )
        return None
    try:
        client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        # Optional: Test connection/bucket existence here if needed
        # client.bucket_exists(MINIO_BUCKET)
        logger.info(
            f"MinIO client created for endpoint: {MINIO_ENDPOINT}, bucket: {MINIO_BUCKET}"
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create MinIO client: {e}", exc_info=True)
        return None


def download_content_minio(file_path: str) -> Optional[bytes]:
    """Downloads raw file content from MinIO."""
    logger.debug(f"Attempting to download from MinIO: {file_path}")
    client = get_minio_client()
    if not client:
        logger.error("MinIO client not available. Cannot download.")
        return None
    try:
        data = client.get_object(MINIO_BUCKET, file_path)
        content = data.read()
        logger.debug(f"Successfully downloaded MinIO file: {file_path}")
        return content
    except S3Error as e:
        logger.error(f"MinIO S3 error downloading {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing {file_path} from MinIO: {e}", exc_info=True)
        return None


def download_content_pcache(relative_file_path: str) -> Optional[bytes]:
    """Downloads raw file content using pcache_fileio."""
    if not PCACHE_AVAILABLE or not OSS_PCACHE_ROOT_DIR:
        logger.error("pcache_fileio not available or OSS_PCACHE_ROOT_DIR not set.")
        return None

    full_file_path = os.path.join(OSS_PCACHE_ROOT_DIR, relative_file_path)
    logger.debug(f"Attempting to download via pcache_fileio: {full_file_path}")
    try:
        with fileio.file_io_impl.open(full_file_path, "rb") as fd:
            content = fd.read()
        logger.debug(f"Successfully downloaded pcache file: {relative_file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"File not found via pcache_fileio: {full_file_path}")
        return None
    except Exception as e:
        logger.error(
            f"Error processing file {relative_file_path} using pcache_fileio: {e}",
            exc_info=True,
        )
        return None


def load_image_from_bytes(content: bytes) -> Optional[np.ndarray]:
    """Decodes image bytes into a NumPy array (RGB)."""
    try:
        img_buffer = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cv2.imdecode failed. Check if content is a valid image.")
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    except Exception as e:
        logger.error(f"Error decoding image bytes: {e}", exc_info=True)
        return None


def load_json_from_bytes(content: bytes, file_identifier: str) -> Optional[Dict]:
    """Parses JSON bytes into a dictionary."""
    try:
        ocr_data = json.loads(content.decode("utf-8"))
        # Basic validation
        if "document" not in ocr_data or not isinstance(ocr_data["document"], list):
            logger.warning(
                f"Invalid OCR JSON format for {file_identifier}: missing 'document' list."
            )
            return None
        return ocr_data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OCR JSON for {file_identifier}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error parsing OCR JSON for {file_identifier}: {e}",
            exc_info=True,
        )
        return None


# === Image/Mask Preprocessing & Generation ===


def get_transforms(resolution: int) -> Tuple[alb.Compose, alb.Compose, alb.Compose]:
    """Creates Albumentations pipelines for image, mask, and tensor conversion."""
    # Transform for original/masked images: Resize, Normalize [-1, 1]
    image_resize_norm = alb.Compose(
        [
            alb.Resize(resolution, resolution, interpolation=cv2.INTER_LANCZOS4),
            alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    # Transform for masks: Resize only
    mask_resize = alb.Compose(
        [
            alb.Resize(resolution, resolution, interpolation=cv2.INTER_NEAREST),
        ]
    )
    # Transform to convert NumPy HWC to PyTorch CHW Tensor
    to_tensor = alb.Compose(
        [
            ToTensorV2(),  # Handles scaling [0, 255] -> [0, 1] or preserves [-1, 1]
        ]
    )
    return image_resize_norm, mask_resize, to_tensor


def draw_text_on_image(text: str, font_path: str) -> np.ndarray:
    """Renders text onto a white background image."""
    font_size = 40
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logger.error(
            f"Font file '{font_path}' not found or invalid. Cannot render text."
        )
        # Return a small dummy black image as fallback
        return np.zeros((60, 100, 3), dtype=np.uint8)

    if not text:
        logger.warning("Attempting to draw empty text. Creating a small blank image.")
        return np.full((60, 100, 3), 255, dtype=np.uint8)  # White image

    # Estimate text size
    try:
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        img_width = text_width + 80  # Padding
        img_height = text_height + 20  # Padding
        pos = (40, 10)  # Top-left position
    except AttributeError:  # Fallback for older PIL
        logger.warning("Using fallback text size estimation (PIL < 9.0.0).")
        img_width = (len(text) + 4) * (font_size // 2)  # Rough estimate
        img_height = font_size + 40
        pos = (20, 20)

    img = Image.new(
        "RGB", (max(img_width, 1), max(img_height, 1)), color="white"
    )  # Ensure min size 1x1
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill="black")

    return np.array(img)  # Return RGB NumPy array


def process_bounding_box(
    location: List[float], image_shape: Tuple[int, int]
) -> List[int]:
    """Adjusts bounding box slightly and ensures it's within image bounds."""
    img_height, img_width = image_shape
    x_min, y_min, x_max, y_max = location

    # Example adjustment: Expand height slightly (optional)
    # h = y_max - y_min
    # y_max_new = min(y_max + h * 0.1, img_height - 1)
    y_max_new = y_max  # No expansion in this version

    # Clamp coordinates and ensure validity
    x_min_int = max(0, int(x_min))
    y_min_int = max(0, int(y_min))
    x_max_int = min(img_width - 1, int(x_max))
    y_max_int = min(img_height - 1, int(y_max_new))

    # Ensure min < max
    if x_max_int <= x_min_int:
        x_max_int = x_min_int + 1
    if y_max_int <= y_min_int:
        y_max_int = y_min_int + 1

    # Clamp again to ensure they don't exceed boundaries after +1 adjustments
    x_max_int = min(img_width - 1, x_max_int)
    y_max_int = min(img_height - 1, y_max_int)

    return [x_min_int, y_min_int, x_max_int, y_max_int]


def generate_mask_from_box(mask_shape: Tuple[int, int], box: List[int]) -> np.ndarray:
    """Generates a binary mask (0/255) from a bounding box."""
    width, height = mask_shape
    mask = Image.new("L", (width, height), 0)  # Black background
    draw = ImageDraw.Draw(mask)
    x_min, y_min, x_max, y_max = box

    # Clamp coordinates just in case
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    if x_max > x_min and y_max > y_min:
        draw.rectangle((x_min, y_min, x_max, y_max), fill=255)  # White rectangle
    else:
        logger.warning(
            f"Invalid rectangle dimensions for mask: {box} within shape {(width, height)}"
        )

    return np.array(mask)  # Return HW NumPy array (0 or 255)


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applies a mask to an image (sets masked area to 0)."""
    # Mask is 0/255, image is RGB. Need boolean mask.
    boolean_mask = mask < 128  # True where mask is 0 (background)
    # Expand mask to 3 channels
    boolean_mask_3c = np.stack([boolean_mask] * image.shape[2], axis=-1)
    masked_image = image * boolean_mask_3c
    return masked_image


# === Custom PyTorch Dataset ===
class DocumentInpaintingDataset(Dataset):
    """
    Dataset for document inpainting. Loads images and OCR, generates masks,
    crops regions, renders text, and applies transformations.
    """

    def __init__(
        self,
        data_csv_path: str,
        ocr_root: str,
        resolution: int,
        crop_scale: int,
        transform_resize_norm: alb.Compose,
        transform_mask_resize: alb.Compose,
        transform_to_tensor: alb.Compose,
        data_source: str = "pcache",
        font_path: str = DEFAULT_FONT_PATH,
        ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
    ):
        """
        Initializes the dataset.

        Args:
            data_csv_path: Path to the CSV file with image paths.
            ocr_root: Root path/prefix for OCR JSON files.
            resolution: Target resolution for final image/mask tensors.
            crop_scale: Size of the square crop region around the text.
            transform_resize_norm: Albumentations transform for resizing and normalization.
            transform_mask_resize: Albumentations transform for resizing masks.
            transform_to_tensor: Albumentations transform to convert to tensor.
            data_source: 'local', 'minio', or 'pcache'.
            font_path: Path to the TTF font file.
            ocr_confidence_threshold: Minimum OCR confidence score.
        """
        self.resolution = resolution
        self.crop_scale = crop_scale
        self.data_csv_path = data_csv_path
        self.ocr_root = ocr_root
        self.data_source = data_source
        self.font_path = font_path
        self.ocr_confidence_threshold = ocr_confidence_threshold

        self.transform_resize_norm = transform_resize_norm
        self.transform_mask_resize = transform_mask_resize
        self.transform_to_tensor = transform_to_tensor

        self.image_paths = self._load_image_paths()
        self.num_images = len(self.image_paths)

        if not self.image_paths:
            raise ValueError(
                f"No image paths loaded from {self.data_csv_path}. Check the file and 'path' column."
            )
        logger.info(
            f"Initialized dataset with {self.num_images} images from {self.data_source}."
        )

    def _load_image_paths(self) -> List[str]:
        """Loads image paths from the CSV file."""
        logger.info(f"Loading image paths from: {self.data_csv_path}")
        try:
            df = pd.read_csv(self.data_csv_path, low_memory=False)
            if "path" not in df.columns:
                raise ValueError(
                    f"CSV file '{self.data_csv_path}' must contain a 'path' column."
                )
            return df["path"].tolist()
        except FileNotFoundError:
            logger.error(f"Training data CSV file not found: {self.data_csv_path}")
            raise
        except Exception as e:
            logger.error(
                f"Error loading image paths from {self.data_csv_path}: {e}",
                exc_info=True,
            )
            raise

    def _load_data(self, relative_path: str) -> Optional[bytes]:
        """Loads raw file content based on the configured data source."""
        if self.data_source == "local":
            # Assuming relative_path is relative to some base directory or absolute
            # This needs clarification based on how local paths are stored in the CSV
            # For now, assume it's directly usable or needs a prefix
            local_path = relative_path  # Adjust if needed, e.g., os.path.join(BASE_DIR, relative_path)
            if os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading local file {local_path}: {e}")
                    return None
            else:
                logger.error(f"Local file not found: {local_path}")
                return None
        elif self.data_source == "minio":
            return download_content_minio(relative_path)
        elif self.data_source == "pcache":
            return download_content_pcache(relative_path)
        else:
            logger.error(f"Unsupported data_source: {self.data_source}")
            return None

    def _get_ocr_path(self, image_path: str) -> str:
        """Constructs the path/key for the OCR JSON file."""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Assume OCR files are under ocr_root with the same base name
        # Adjust if the structure is different (e.g., different subdirs)
        return os.path.join(self.ocr_root, base_name + ".json")

    def _process_ocr_data(
        self, ocr_data: Dict, image_path: str
    ) -> Optional[pd.DataFrame]:
        """Parses OCR JSON and filters by confidence."""
        try:
            ocr_pd = pd.DataFrame(ocr_data["document"])
            if not all(col in ocr_pd.columns for col in ["text", "score", "box"]):
                logger.warning(
                    f"OCR data frame missing required columns for {image_path}"
                )
                return None
            ocr_pd_filtered = ocr_pd[
                ocr_pd["score"] > self.ocr_confidence_threshold
            ].copy()
            if ocr_pd_filtered.empty:
                logger.warning(f"No high-confidence OCR results found for {image_path}")
                return None
            return ocr_pd_filtered
        except Exception as e:
            logger.error(
                f"Error processing OCR data for {image_path}: {e}", exc_info=True
            )
            return None

    def _get_crop_coordinates(
        self, box: List[int], image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculates random crop start coordinates (x_s, y_s) containing the box."""
        x1, y1, x2, y2 = box
        current_h, current_w = image_shape
        _box_w, _box_h = x2 - x1, y2 - y1

        # Calculate valid start ranges for the crop window
        min_start_x = max(0, x2 - self.crop_scale)  # Must end after box ends
        max_start_x = min(
            current_w - self.crop_scale, x1
        )  # Must start before box starts
        min_start_y = max(0, y2 - self.crop_scale)
        max_start_y = min(current_h - self.crop_scale, y1)

        # Clamp ranges to ensure they are valid
        min_start_x = min(min_start_x, current_w - self.crop_scale)
        max_start_x = max(max_start_x, 0)
        min_start_y = min(min_start_y, current_h - self.crop_scale)
        max_start_y = max(max_start_y, 0)

        # Randomly select start coordinates
        try:
            x_s = (
                np.random.randint(min_start_x, max_start_x + 1)
                if max_start_x >= min_start_x
                else min_start_x
            )
            y_s = (
                np.random.randint(min_start_y, max_start_y + 1)
                if max_start_y >= min_start_y
                else min_start_y
            )
        except ValueError as ive:
            logger.warning(
                f"Issue calculating crop start: {ive}. Using fallback. Box: {box}, Img Size: {image_shape}"
            )
            x_s = max(
                0, current_w - self.crop_scale
            )  # Fallback: crop from bottom-right
            y_s = max(0, current_h - self.crop_scale)

        return int(x_s), int(y_s)

    def __getitem__(self, index):
        """Loads image/OCR, processes, crops, transforms, and returns tensors."""
        image_path = self.image_paths[index]
        ocr_path = self._get_ocr_path(image_path)
        example = {}

        try:
            # 1. Load Image and OCR Data
            image_content = self._load_data(image_path)
            ocr_content = self._load_data(ocr_path)

            if image_content is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            if ocr_content is None:
                raise FileNotFoundError(f"OCR not found: {ocr_path}")

            instance_image = load_image_from_bytes(image_content)  # RGB NumPy array
            ocr_data = load_json_from_bytes(ocr_content, ocr_path)

            if instance_image is None:
                raise ValueError("Image decoding failed")
            if ocr_data is None:
                raise ValueError("OCR JSON parsing failed")

            original_shape_hw = instance_image.shape[:2]  # (H, W)

            # 2. Process OCR and Sample One Detection
            ocr_pd_filtered = self._process_ocr_data(ocr_data, image_path)
            if ocr_pd_filtered is None:
                raise ValueError("No valid OCR data found")

            ocr_sample = ocr_pd_filtered.sample(n=1).iloc[0]
            text = str(ocr_sample["text"])
            try:
                box_coords = ocr_sample["box"]
                x_coords = [p[0] for p in box_coords]
                y_coords = [p[1] for p in box_coords]
                raw_location = [
                    min(x_coords),
                    min(y_coords),
                    max(x_coords),
                    max(y_coords),
                ]
            except (IndexError, TypeError):
                raise ValueError(f"Invalid 'box' format in OCR data: {box_coords}")

            location_box = process_bounding_box(
                raw_location, original_shape_hw
            )  # [x_min, y_min, x_max, y_max] ints

            # 3. Generate Mask and Masked Image (Original Size)
            mask = generate_mask_from_box(
                (original_shape_hw[1], original_shape_hw[0]), location_box
            )  # HW, 0/255
            masked_image = apply_mask_to_image(instance_image, mask)  # RGB

            # 4. Handle Resizing if Image < Crop Scale (Optional Pre-Crop Resize)
            h, w = original_shape_hw
            if min(h, w) < self.crop_scale:
                scale_factor = self.crop_scale / min(h, w)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                instance_image = cv2.resize(
                    instance_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
                )
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                masked_image = cv2.resize(
                    masked_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
                )
                # Scale box coordinates
                location_box = [int(c * scale_factor) for c in location_box]
                # Clamp again after scaling
                location_box = process_bounding_box(location_box, (new_h, new_w))
                current_shape_hw = (new_h, new_w)
            else:
                current_shape_hw = original_shape_hw

            # 5. Determine Crop Coordinates
            x_s, y_s = self._get_crop_coordinates(location_box, current_shape_hw)

            # 6. Crop Image, Mask, Masked Image
            instance_image_crop = instance_image[
                y_s : y_s + self.crop_scale, x_s : x_s + self.crop_scale, :
            ]
            mask_crop = mask[y_s : y_s + self.crop_scale, x_s : x_s + self.crop_scale]
            masked_image_crop = masked_image[
                y_s : y_s + self.crop_scale, x_s : x_s + self.crop_scale, :
            ]

            # 7. Render Text
            draw_ttf = draw_text_on_image(text, self.font_path)  # RGB NumPy

            # 8. Apply Final Transformations (Resize to `resolution`, Normalize, ToTensor)
            # Apply resize/norm to cropped original image
            augmented = self.transform_resize_norm(image=instance_image_crop)
            instance_image_final = augmented["image"]  # HWC, [-1, 1]
            augmented = self.transform_to_tensor(image=instance_image_final)
            example["pixel_values"] = augmented["image"]  # CHW Tensor, [-1, 1]

            # Apply resize/norm to cropped masked image
            augmented = self.transform_resize_norm(image=masked_image_crop)
            masked_image_final = augmented["image"]  # HWC, [-1, 1]
            augmented = self.transform_to_tensor(image=masked_image_final)
            example["masked_image"] = augmented["image"]  # CHW Tensor, [-1, 1]

            # Apply resize to cropped mask
            augmented = self.transform_mask_resize(image=mask_crop)  # HW, 0/255
            mask_final = augmented["image"]
            # Convert mask to float tensor [0, 1]
            mask_final = mask_final.astype(np.float32) / 255.0
            augmented = self.transform_to_tensor(
                image=mask_final[:, :, np.newaxis]
            )  # Add channel dim
            example["mask"] = augmented["image"]  # CHW Tensor (C=1), [0, 1]

            # Convert rendered text image to tensor (TrOCR processor handles normalization later)
            augmented = self.transform_to_tensor(image=draw_ttf)  # CHW Tensor, [0, 1]
            example["ttf_image"] = augmented["image"]

            example["is_valid"] = True  # Mark as valid sample
            return example

        except Exception as e:
            logger.error(
                f"Error processing data at index {index} (path: {image_path}): {e}",
                exc_info=True,
            )
            # Return dummy data to avoid crashing batch
            dummy_img = torch.zeros(3, self.resolution, self.resolution)
            dummy_mask = torch.zeros(1, self.resolution, self.resolution)
            dummy_ttf = torch.zeros(3, 40, 100)  # Placeholder size
            return {
                "pixel_values": dummy_img,
                "masked_image": dummy_img,
                "mask": dummy_mask,
                "ttf_image": dummy_ttf,
                "is_valid": False,  # Mark as invalid
            }

    def __len__(self):
        return self.num_images


# === Collate Function ===
def collate_fn_doc_inpainting(examples):
    """
    Custom collate function. Filters invalid examples and stacks tensors.
    Returns None if the entire batch is invalid.
    """
    valid_examples = [ex for ex in examples if ex["is_valid"]]
    if not valid_examples:
        logger.warning("Entire batch is invalid. Skipping.")
        return None

    # Stack tensors for valid examples
    pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])
    masked_images = torch.stack([example["masked_image"] for example in valid_examples])
    masks = torch.stack([example["mask"] for example in valid_examples])
    # TTF images might have variable sizes, keep as list for TrOCR processor
    ttf_images = [example["ttf_image"] for example in valid_examples]

    # Ensure contiguous float tensors
    batch = {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "masked_images": masked_images.to(
            memory_format=torch.contiguous_format
        ).float(),
        "masks": masks.to(memory_format=torch.contiguous_format).float(),
        "ttf_images": ttf_images,  # List of tensors
    }
    return batch


# === Hugging Face Hub Utilities ===
def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
) -> str:
    """Constructs the full repository name for the Hugging Face Hub."""
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        if token is None:
            raise ValueError(
                "No organization provided and no token found. Please login using `huggingface-cli login` or pass --hub_token."
            )
        try:
            username = whoami(token)["name"]
            return f"{username}/{model_id}"
        except Exception as e:
            raise ValueError(f"Could not get username from token. Error: {e}")
    else:
        return f"{organization}/{model_id}"


# === Tensor/NumPy Conversion (Optional, for debugging/visualization) ===
def tensor2im(input_image: torch.Tensor, imtype=np.uint8) -> np.ndarray:
    """Converts a PyTorch Tensor [-1, 1] to a NumPy image [0, 255]."""
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image  # Or raise error
        image_numpy = (
            image_tensor[0].cpu().float().numpy()
        )  # Select first image if batch
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # CHW -> HWC
        image_numpy = (image_numpy + 1) / 2.0 * 255.0  # Denormalize [-1, 1] -> [0, 255]
    else:
        image_numpy = input_image  # Assume already HWC [0, 255] if NumPy

    return np.clip(image_numpy, 0, 255).astype(imtype)


# === Environment Setup ===
def setup_environment(args, accelerator):
    """Sets up logging, seeding, and TF32."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Set verbosity for underlying libraries
    if accelerator.is_local_main_process:
        datasets_logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets_logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("Enabled TF32 for matmul.")


# === Model Loading ===
def load_models_and_scheduler(args):
    """Loads VAE, UNet, TrOCR, Noise Scheduler, and TrOCR Processor."""
    logger.info("Loading models and scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=args.cache_dir,
    )
    processor = TrOCRProcessor.from_pretrained(
        args.trocr_model_name, cache_dir=args.cache_dir
    )
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        args.trocr_model_name, cache_dir=args.cache_dir
    ).encoder
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision
        or args.revision,  # Use non_ema if specified, else main revision
        cache_dir=args.cache_dir,
    )
    logger.info("Models and scheduler loaded.")
    return noise_scheduler, processor, trocr_model, vae, unet


# === Optimizer and LR Scheduler Setup ===
def setup_optimizer_and_scheduler(args, unet, accelerator):
    """Initializes the optimizer and learning rate scheduler."""
    logger.info("Setting up optimizer and LR scheduler...")
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer.")
        except ImportError:
            logger.warning("bitsandbytes not found. Falling back to standard AdamW.")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
        logger.info("Using standard AdamW optimizer.")

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # LR Scheduler setup needs total steps, calculated later after dataloader prep
    logger.info(
        "Optimizer initialized. LR scheduler will be created after dataloader preparation."
    )
    return optimizer


# === Data Preparation ===
def prepare_datasets(args):
    """Creates the dataset and dataloader."""
    logger.info("Preparing dataset and dataloader...")
    image_resize_norm, mask_resize, to_tensor = get_transforms(args.resolution)
    train_dataset = DocumentInpaintingDataset(
        data_csv_path=args.train_data_csv,
        ocr_root=args.ocr_data_root,
        resolution=args.resolution,
        crop_scale=args.crop_scale,
        transform_resize_norm=image_resize_norm,
        transform_mask_resize=mask_resize,
        transform_to_tensor=to_tensor,
        data_source=args.data_source,
        font_path=args.font_path,
        ocr_confidence_threshold=args.ocr_confidence_threshold,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn_doc_inpainting,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    logger.info("Dataset and Dataloader created.")
    return train_dataset, train_dataloader


# === Accelerator Hooks for Diffusers Saving ===
def create_accelerator_hooks(args, ema_unet, accelerator):
    """Creates and registers save/load hooks for Accelerator."""
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            logger.debug(f"Save hook called for {output_dir}")
            if args.use_ema and ema_unet is not None:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                logger.info(f"Saved EMA UNet to {os.path.join(output_dir, 'unet_ema')}")

            if models:  # Should contain the UNet
                models[0].save_pretrained(os.path.join(output_dir, "unet"))
                logger.info(f"Saved UNet to {os.path.join(output_dir, 'unet')}")
                # Pop weights to prevent Accelerate from saving them again
                if weights:
                    weights.pop()  # Check if weights list is not empty

        def load_model_hook(models, input_dir):
            logger.debug(f"Load hook called for {input_dir}")
            if args.use_ema and ema_unet is not None:
                logger.info(
                    f"Loading EMA UNet from {os.path.join(input_dir, 'unet_ema')}"
                )
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
                logger.info("EMA UNet loaded.")

            if models:  # Should contain the UNet placeholder
                logger.info(f"Loading UNet from {os.path.join(input_dir, 'unet')}")
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                models[0].register_to_config(
                    **load_model.config
                )  # Use the first model in the list
                models[0].load_state_dict(load_model.state_dict())
                del load_model
                logger.info("UNet loaded.")
            # No need to pop models here, Accelerate handles it

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        logger.info("Registered Accelerator save/load hooks for Diffusers format.")
    else:
        logger.warning(
            "Accelerator version < 0.16.0. Custom save/load hooks not registered. Checkpoint saving might use standard PyTorch format."
        )


# === Training Step ===
def train_step(
    batch, unet, vae, trocr_model, processor, noise_scheduler, weight_dtype, accelerator
):
    """Performs a single training step including forward pass, loss calculation, and returns loss."""
    # --- a. Prepare Text Conditioning ---
    try:
        # TrOCR processor expects PIL Images or List[torch.Tensor] or pixel_values
        # Our collate_fn returns List[CHW Tensor [0, 1]]
        # Convert to List[PIL Image] or ensure processor handles tensors correctly
        # Assuming processor handles list of tensors:
        processed_text_input = processor(
            images=batch["ttf_images"], return_tensors="pt"
        ).pixel_values
    except Exception as proc_e:
        logger.error(f"Error during TrOCR processing: {proc_e}", exc_info=True)
        raise ValueError("TrOCR processing failed") from proc_e  # Propagate error

    processed_text_input = processed_text_input.to(
        accelerator.device, dtype=weight_dtype
    )

    with torch.no_grad():
        ocr_feature = trocr_model(processed_text_input)
        ocr_embeddings = (
            ocr_feature.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

    # --- b. Prepare Image Latents and Mask ---
    with torch.no_grad():
        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype)
        ).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        masked_image_latents = vae.encode(
            batch["masked_images"].to(dtype=weight_dtype)
        ).latent_dist.sample()
        masked_image_latents = masked_image_latents * vae.config.scaling_factor

    # Prepare mask for latent space
    mask = F.interpolate(
        batch["masks"].to(
            accelerator.device, dtype=weight_dtype
        ),  # Ensure mask is on device
        size=(latents.shape[2], latents.shape[3]),
    )

    # --- c. Prepare Noisy Latents ---
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # --- d. UNet Prediction ---
    # Concatenate inputs: noisy_latents, mask, masked_image_latents
    model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
    encoder_hidden_states = ocr_embeddings.detach()  # Ensure detached

    model_pred = unet(model_input, timesteps, encoder_hidden_states).sample

    # --- e. Calculate Loss ---
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
        )

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss


# === Main Training Loop ===
def training_loop(
    args,
    accelerator,
    unet,
    vae,
    trocr_model,
    processor,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    train_dataloader,
    ema_unet,
    weight_dtype,
):
    """The main training loop."""

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Starting Training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    # --- Resume from Checkpoint ---
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:  # Find latest checkpoint
            dirs = [
                d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")
            ]
            if not dirs:
                logger.warning(
                    "Resume 'latest' requested, but no checkpoint found. Starting fresh."
                )
                args.resume_from_checkpoint = None  # Disable resume
                path = None
            else:
                dirs.sort(key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]

        if path and os.path.isdir(os.path.join(args.output_dir, path)):
            resume_path = os.path.join(args.output_dir, path)
            logger.info(f"Resuming from checkpoint {resume_path}")
            try:
                accelerator.load_state(resume_path)
                global_step = int(path.split("-")[1])
                num_update_steps_per_epoch = math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = global_step % num_update_steps_per_epoch
                logger.info(
                    f"Resumed training from global step {global_step}, epoch {first_epoch}, step {resume_step}."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load checkpoint state from {resume_path}: {e}. Starting fresh."
                )
                global_step = 0
                first_epoch = 0
                resume_step = 0
                args.resume_from_checkpoint = None  # Disable resume on failure
        else:
            logger.warning(
                f"Checkpoint '{args.resume_from_checkpoint}' not found or invalid. Starting fresh."
            )
            args.resume_from_checkpoint = None

    # --- Progress Bar ---
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps",
    )

    # --- Epoch and Step Loop ---
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss_accum = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps before resume point
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # Handle invalid batches from collate_fn
            if batch is None:
                logger.warning(
                    f"Skipping step {step} in epoch {epoch} due to invalid batch."
                )
                # Need to potentially update progress bar if this skip causes a missed optimizer step count
                # This logic is tricky; simplest is to just continue.
                continue

            # --- Forward & Backward Pass ---
            with accelerator.accumulate(unet):
                try:
                    loss = train_step(
                        batch,
                        unet,
                        vae,
                        trocr_model,
                        processor,
                        noise_scheduler,
                        weight_dtype,
                        accelerator,
                    )
                except (
                    ValueError
                ) as e:  # Catch errors from train_step (e.g., TrOCR failure)
                    logger.error(f"Skipping step {step} due to error: {e}")
                    # If using gradient accumulation, need to ensure optimizer step isn't skipped incorrectly.
                    # Simplest: continue, let accumulate handle sync.
                    continue

                # Gather loss across devices for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss_accum += avg_loss.item() / args.gradient_accumulation_steps

                # Backward pass
                accelerator.backward(loss)

                # Optimizer step (includes gradient clipping if needed)
                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:  # Only clip if max_grad_norm is positive
                        accelerator.clip_grad_norm_(
                            unet.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # --- Post-Step Operations ---
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1

                # Logging
                if (
                    global_step % args.checkpointing_steps == 0
                ):  # Log less frequently than every step
                    logs = {
                        "step_loss": train_loss_accum,  # Log accumulated loss
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log({"train_loss": train_loss_accum}, step=global_step)
                    train_loss_accum = 0.0  # Reset accumulator
                else:  # Log instantaneous loss more frequently if needed
                    logs = {
                        "inst_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)  # Uses custom hooks
                        logger.info(f"Saved checkpoint state to {save_path}")

                        # Optional: Save pipeline for easy testing (consider helper function)
                        # save_pipeline(args, accelerator, unet, vae, ema_unet, weight_dtype, save_path)

            # Check for exit condition
            if global_step >= args.max_train_steps:
                logger.info("Reached max_train_steps. Exiting training loop.")
                break  # Exit inner loop

        # Check again after epoch end
        if global_step >= args.max_train_steps:
            break  # Exit outer loop

    # --- End of Training ---
    logger.info("Training finished.")
    accelerator.wait_for_everyone()


# === Saving Final Model and Pipeline ===
def save_final_model(args, accelerator, unet, vae, ema_unet, weight_dtype):
    """Saves the final trained model and optionally the pipeline."""
    if accelerator.is_main_process:
        unet_final = accelerator.unwrap_model(unet)
        if args.use_ema:
            logger.info("Copying EMA weights to final UNet model.")
            ema_unet.copy_to(unet_final.parameters())

        # Save UNet in diffusers format
        unet_final.save_pretrained(os.path.join(args.output_dir, "unet"))
        logger.info(
            f"Saved final UNet model to {os.path.join(args.output_dir, 'unet')}"
        )

        # Save the full pipeline (requires text_encoder/tokenizer from base model)
        # This part assumes the base model IS a standard Stable Diffusion model
        # If not, saving a pipeline might not be directly applicable or require modification
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet_final,
                vae=accelerator.unwrap_model(vae),  # Ensure VAE is unwrapped
                revision=args.revision,
                torch_dtype=weight_dtype,
                cache_dir=args.cache_dir,
                # Requires text_encoder & tokenizer from the base model for standard pipeline saving
            )
            pipeline.save_pretrained(args.output_dir)
            logger.info(f"Saved final pipeline to {args.output_dir}")
        except Exception as e:
            logger.warning(
                f"Could not save final pipeline (may require components like text_encoder): {e}"
            )

        # Clean up .gitignore if pushing to hub
        if args.push_to_hub:
            gitignore_path = os.path.join(args.output_dir, ".gitignore")
            if os.path.exists(gitignore_path):
                try:
                    with open(gitignore_path, "r") as f:
                        lines = f.readlines()
                    with open(gitignore_path, "w") as f:
                        for line in lines:
                            if not line.strip().startswith(
                                ("step_*", "epoch_*", "checkpoint-*")
                            ):
                                f.write(line)
                    logger.info("Cleaned .gitignore for Hub push.")
                except Exception as e:
                    logger.warning(f"Could not clean .gitignore: {e}")


# === Main Execution ===
def main():
    args = parse_args()

    # --- Accelerator Setup ---
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

    # --- Environment Setup ---
    setup_environment(args, accelerator)

    # --- Repository Setup (Optional) ---
    repo = None
    repo_name = None
    if accelerator.is_main_process and args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(args.output_dir).name,
                organization=args.hub_organization,
                token=args.hub_token,
            )
        else:
            repo_name = args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=args.hub_token)
        repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)
        # Add checkpoint patterns to .gitignore
        with open(os.path.join(args.output_dir, ".gitignore"), "a+") as gitignore:
            gitignore.seek(0)
            content = gitignore.read()
            if "checkpoint-*/" not in content:
                gitignore.write("checkpoint-*/\n")
            # Add other patterns if needed (step_*, epoch_*)

    # --- Load Models ---
    noise_scheduler, processor, trocr_model, vae, unet = load_models_and_scheduler(args)

    # --- Freeze Parameters ---
    vae.requires_grad_(False)
    trocr_model.requires_grad_(False)
    logger.info("Froze VAE and TrOCR parameters.")

    # --- EMA Setup ---
    ema_unet = None
    if args.use_ema:
        # EMA model setup needs to happen *before* accelerator.prepare
        # It wraps the *parameters* of the model being trained.
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )
        logger.info("EMA model initialized.")

    # --- Accelerator Hooks ---
    # Create hooks *before* prepare, passing the ema_unet instance if used
    create_accelerator_hooks(args, ema_unet, accelerator)

    # --- Optimizations ---
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing.")
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention.")
            except Exception as e:
                logger.warning(
                    f"Could not enable xformers: {e}. Proceeding without it."
                )
        else:
            logger.warning(
                "xformers not available. Install for potential memory savings."
            )

    # --- Optimizer ---
    optimizer = setup_optimizer_and_scheduler(args, unet, accelerator)

    # --- Data ---
    train_dataset, train_dataloader = prepare_datasets(args)

    # --- LR Scheduler (needs total steps) ---
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(f"Calculated max_train_steps: {args.max_train_steps}")
    else:
        # Recalculate epochs if max_train_steps is set
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
        logger.info(
            f"Using provided max_train_steps: {args.max_train_steps}. Effective epochs: {args.num_train_epochs}"
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --- Prepare with Accelerator ---
    logger.info("Preparing components with Accelerator...")
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    # Move EMA model to device *after* prepare
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # --- Prepare Inference Models (VAE, TrOCR) ---
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    trocr_model.to(accelerator.device, dtype=weight_dtype)
    logger.info(
        f"Moved VAE and TrOCR to {accelerator.device} with dtype {weight_dtype}"
    )

    # --- Initialize Trackers ---
    if accelerator.is_main_process:
        accelerator.init_trackers("diffute_doc_inpainting", config=vars(args))
        logger.info("Initialized trackers.")

    # --- Training ---
    training_loop(
        args,
        accelerator,
        unet,
        vae,
        trocr_model,
        processor,
        noise_scheduler,
        optimizer,
        lr_scheduler,
        train_dataloader,
        ema_unet,
        weight_dtype,
    )

    # --- Save Final Model & Push to Hub ---
    save_final_model(args, accelerator, unet, vae, ema_unet, weight_dtype)

    if accelerator.is_main_process and args.push_to_hub and repo is not None:
        logger.info("Pushing final model to Hugging Face Hub...")
        repo.push_to_hub(
            commit_message="End of training", blocking=False, auto_lfs_prune=True
        )
        logger.info(f"Model pushed to Hub repository: {repo_name}")

    accelerator.end_training()
    logger.info("Script finished.")


if __name__ == "__main__":
    main()
