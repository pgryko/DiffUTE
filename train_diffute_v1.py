"""
DiffUTE (Diffusion Universal Text Editor) Training Script

This script implements the training process for the DiffUTE model, which is designed
to edit text in images while preserving the surrounding context. The model combines
a pre-trained VAE with a UNet-based diffusion model and TrOCR for text understanding.

Architecture Overview:
--------------------
1. VAE (Variational AutoEncoder):
   - Pre-trained and frozen during this training
   - Handles image encoding/decoding
   - Reduces computational complexity by working in latent space
   - Converts images between pixel space (H×W×3) and latent space (h×w×4)

2. UNet:
   - Main component being trained
   - Performs denoising in latent space
   - Conditioned on text embeddings from TrOCR
   - Takes concatenated input of:
     * Noisy latents
     * Binary mask
     * Masked image latents

3. TrOCR:
   - Pre-trained text recognition model
   - Provides text embeddings for conditioning
   - Frozen during training
   - Helps guide the text generation process

Training Process:
---------------
1. Data Preparation:
   - Original images are encoded to latent space using VAE
   - Text regions are masked
   - Target text is rendered and processed by TrOCR
   - All inputs are normalized to [-1, 1] range

2. Forward Diffusion:
   - Add random noise to latent representations
   - Sample random timesteps for each batch
   - Apply noise schedule based on timesteps
   - Create noisy versions of the input

3. Reverse Diffusion Training:
   - UNet learns to predict noise or velocity
   - Guided by:
     * Text embeddings from TrOCR
     * Masked regions indicating edit areas
     * Original image context
   - Uses MSE loss between predictions and targets

4. Conditioning Strategy:
   - Text embeddings provide semantic guidance
   - Masks ensure localized editing
   - Original context helps maintain image coherence

Key Features:
-----------
1. Multi-modal Integration:
   - Combines image and text understanding
   - Uses cross-attention for text conditioning
   - Maintains spatial awareness through masks

2. Controlled Generation:
   - Precise text region targeting
   - Context preservation outside edit areas
   - Style-aware text generation

3. Training Optimizations:
   - Mixed precision training
   - Gradient accumulation
   - Distributed training support
   - Checkpoint management

Usage:
------
python train_diffute_v1.py
    --pretrained_model_name_or_path <path>
    --output_dir <dir>
    --train_batch_size 16
    --num_train_epochs 100
    --learning_rate 1e-4

Dependencies:
------------
- PyTorch: Deep learning framework
- Diffusers: Diffusion model implementation
- Transformers: For TrOCR model
- Accelerate: Distributed training support
- Albumentations: Image augmentation

Notes:
-----
- The VAE must be pre-trained and frozen
- TrOCR is used in inference mode only
- Training requires significant GPU memory
- Batch size may need adjustment based on available memory
- Learning rate scheduling is critical for stability
"""

from pcache_fileio import fileio
from pcache_fileio.oss_conf import OssConfigFactory
import pandas as pd
import numpy as np
import os
import cv2
import json
from PIL import Image, ImageDraw, ImageFont, ImageFile

OSS_CONF_NAME = "oss_conf_1"  # Optional, name of your oss configure
OSS_ID = "xxx"  # Your oss id
OSS_KEY = "xxx"  # Your oss key
OSS_ENDPOINT = "cn-heyuan-alipay-office.oss-alipay.aliyuncs.com"  # Your oss endpoint
OSS_BUCKET = "xxx"  # Your oss bucket name
OSS_PCACHE_ROOT_DIR = "oss://" + OSS_BUCKET
OssConfigFactory.register(OSS_ID, OSS_KEY, OSS_ENDPOINT, OSS_CONF_NAME)

import argparse
import logging
import math
from pathlib import Path
from typing import Optional
import accelerate
import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import diffusers
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from alps.pytorch.api.utils.web_access import patch_requests

patch_requests()
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--guidance_scale", type=float, default=0.8)

    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--select_data_lenth",
        type=int,
        default=100,
        help="Number of images selected for training.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


image_trans_resize_and_crop = alb.Compose(
    [
        alb.Resize(512, 512),
        alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

mask_resize_and_crop = alb.Compose(
    [
        alb.Resize(512, 512),
    ]
)

image_trans = alb.Compose(
    [
        ToTensorV2(),
    ]
)


def draw_text(im_shape, text):
    font_size = 40
    font_file = "arialuni.ttf"
    len_text = len(text)
    if len_text == 0:
        len_text = 3
    img = Image.new("RGB", ((len_text + 2) * font_size, 60), color="white")
    # Define the font object
    font = ImageFont.truetype(font_file, font_size)
    # Define the text and position
    pos = (40, 10)

    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill="black")
    img = np.array(img)

    return img


def process_location(location, instance_image_size):
    """
    Process and adjust text region coordinates to include padding.

    Args:
        location (list): [x0, y0, x1, y1] coordinates of text region
        instance_image_size (tuple): (height, width) of the image

    Returns:
        list: Adjusted coordinates with added padding
    """
    h = location[3] - location[1]
    location[3] = min(location[3] + h / 10, instance_image_size[0] - 1)
    return location


def generate_mask(im_shape, ocr_locate):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        (ocr_locate[0], ocr_locate[1], ocr_locate[2], ocr_locate[3]),
        fill=1,
    )
    mask = np.array(mask)
    return mask


def prepare_mask_and_masked_image(image, mask):
    """
    Create a masked version of the input image.

    Args:
        image (numpy.ndarray): Input image array (H, W, C)
        mask (numpy.ndarray): Binary mask array (H, W)

    Returns:
        numpy.ndarray: Image with masked region set to 0
    """
    masked_image = np.multiply(
        image, np.stack([mask < 0.5, mask < 0.5, mask < 0.5]).transpose(1, 2, 0)
    )
    return masked_image


def download_oss_file_pcache(my_file="xxx"):
    MY_FILE_PATH = os.path.join(OSS_PCACHE_ROOT_DIR, my_file)
    with fileio.file_io_impl.open(MY_FILE_PATH, "rb") as fd:
        content = fd.read()
    img = np.frombuffer(content, dtype=np.int8)
    img = cv2.imdecode(img, flags=1)
    return img


class OursDataset(Dataset):
    """
    A dataset to prepare the instance images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        size=512,
        transform_resize_crop=None,
        transform_to_tensor=None,
        mask_transform=None,
    ):
        self.size = size
        self.instance_images_paths, self.ocr_paths = [], []
        self._load_images_paths()
        self.num_instance_images = len(self.instance_images_paths)
        self._length = self.num_instance_images

        self.transform_resize_crop = transform_resize_crop
        self.transform_to_tensor = transform_to_tensor
        self.mask_transform = mask_transform

    def __len__(self):
        return self._length

    def _load_images_paths(self):
        print("loading training file...")
        df = pd.read_csv("doc_select.csv", low_memory=False)
        img_path = df["image_path"]
        ocr_path = df["ocr_path"]
        self.instance_images_paths = img_path.tolist()[:]
        self.ocr_paths = ocr_path.tolist()[:]

        df5 = pd.read_csv("doc.csv", low_memory=False)
        path5 = "diffdoc/" + df5["path"]
        self.path5 = path5.tolist()[:]

    def __getitem__(self, index):
        example = {}
        instance_image = download_oss_file_pcache(self.instance_images_paths[i])
        instance_image_size = instance_image.shape

        # 判断instance image是否存在
        with fileio.file_io_impl.open("xxx" + temp_ocr_path, "r") as f:
            content = f.read()
        ocr_res = json.loads(content)
        ocr_pd = pd.DataFrame(ocr_res["document"])
        ocr_pd = ocr_pd[ocr_pd["score"] > 0.8]
        h, w, c = instance_image.shape

        ocr_pd_sample = ocr_pd.sample()
        instance_image_size = instance_image.shape
        text = ocr_pd_sample["text"].tolist()[0]
        location = ocr_pd_sample["box"].tolist()[0]
        location = list(
            [
                min([x[0] for x in location]),
                min([x[1] for x in location]),
                max([x[0] for x in location]),
                max([x[1] for x in location]),
            ]
        )
        location = process_location(location, instance_image_size)
        location = np.int32(location)
        # 生成mask和masked image
        crop_scale = 256
        mask = generate_mask(instance_image.shape[:2][::-1], location)
        masked_image = prepare_mask_and_masked_image(instance_image, mask)

        # 判断是否可以crop图片
        short_side = min(h, w)
        if short_side < crop_scale:
            scale_factor = int(crop_scale * 2 / short_side)
            new_h, new_w = h * scale_factor, w * scale_factor
            instance_image = cv2.resize(instance_image, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h))
            masked_image = cv2.resize(masked_image, (new_w, new_h))

        # crop text和对应location
        x1, y1, x2, y2 = location
        if x2 - x1 < crop_scale:
            try:
                x_s = np.random.randint(max(0, x2 - crop_scale), x1)
            except:
                x_s = 0
            text = text
        else:
            x_s = x1
            text = text[: int(len(text) * (crop_scale) / (x2 - x1))]
        if y2 - y1 < crop_scale:
            try:
                y_s = np.random.randint(max(0, y2 - crop_scale), y1)
            except:
                y_s = 0
            text = text
        else:
            y_s = y1
            text = text[: int(len(text) * (crop_scale) / (y2 - y1))]

        draw_ttf = draw_text(instance_image.shape[:2][::-1], text)
        instance_image_1 = instance_image[
            y_s : y_s + crop_scale, x_s : x_s + crop_scale, :
        ]
        mask_crop = mask[y_s : y_s + crop_scale, x_s : x_s + crop_scale]
        masked_image_crop = masked_image[
            y_s : y_s + crop_scale, x_s : x_s + crop_scale, :
        ]

        augmented = self.transform_resize_crop(image=instance_image_1)
        instance_image_1 = augmented["image"]
        augmented = self.transform_to_tensor(image=instance_image_1)
        instance_image_1 = augmented["image"]

        augmented = self.transform_resize_crop(image=masked_image_crop)
        masked_image_crop = augmented["image"]
        augmented = self.transform_to_tensor(image=masked_image_crop)
        masked_image_crop = augmented["image"]

        augmented = self.mask_transform(image=mask_crop)
        mask_crop = augmented["image"]
        augmented = self.transform_to_tensor(image=mask_crop)
        mask_crop = augmented["image"]

        augmented = self.transform_to_tensor(image=draw_ttf)
        draw_ttf = augmented["image"]

        example["instance_images"] = instance_image_1
        example["mask"] = mask_crop
        example["masked_image"] = masked_image_crop
        example["ttf_img"] = draw_ttf

        return example


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def numpy_to_pil(images):
    """
    Convert numpy image arrays to PIL images.

    Args:
        images (numpy.ndarray): Image array(s) in range [0, 1]

    Returns:
        list[PIL.Image]: List of PIL images
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def tensor2im(input_image, imtype=np.uint8):
    """
    Convert a tensor to a numpy image array.

    Args:
        input_image (torch.Tensor or numpy.ndarray): Input image
        imtype (type): Desired output numpy dtype

    Returns:
        numpy.ndarray: Processed image array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def main():
    """
    Main training function for the DiffUTE model.

    This function implements the core training loop:
    1. Setup Phase:
       - Initialize models (VAE, UNet, TrOCR)
       - Configure training (optimizer, scheduler, etc.)
       - Prepare data loading

    2. Training Loop:
       - Process batches of images and text
       - Generate noise and conditioning
       - Train UNet for denoising
       - Track and log progress

    3. Model Management:
       - Save checkpoints
       - Handle distributed training
       - Monitor training metrics

    The training process focuses on teaching the UNet to:
    - Understand text through TrOCR embeddings
    - Denoise effectively in masked regions
    - Maintain image context outside text areas
    """
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
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

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            Repository(
                args.output_dir, clone_from=repo_name, token=args.hub_token
            )

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-printed"
    ).encoder
    vae = AutoencoderKL.from_pretrained(
        "./diffdoc-vae-512/checkpoint-350000/", subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )

    # Freeze vae and text_encoder
    trocr_model.requires_grad_(False)
    vae.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def collate_fn_ours(examples):
        pixel_values = torch.stack([example["instance_images"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        masks = []
        masked_images = []
        ttf_imgs = []
        for example in examples:
            masks.append(example["mask"])
            masked_images.append(example["masked_image"])
            ttf_imgs.append(example["ttf_img"])

        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)

        batch = {
            "pixel_values": pixel_values,
            "masks": masks,
            "masked_images": masked_images,
            "ttf_images": ttf_imgs,
        }

        return batch

    # DataLoaders creation:

    datasets_doc = OursDataset(
        size=512,
        transform_resize_crop=image_trans_resize_and_crop,
        transform_to_tensor=image_trans,
        mask_transform=mask_resize_and_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        datasets_doc,
        shuffle=True,
        collate_fn=collate_fn_ours,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    trocr_model.to(accelerator.device, dtype=weight_dtype)

    # Rex: 获取VAE downsample比例
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(datasets_doc)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # guidance_scale = args.guidance_scale
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # Process text with TrOCR for conditioning
            pixel_values = processor(
                images=batch["ttf_images"], return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(accelerator.device, dtype=weight_dtype)
            ocr_feature = trocr_model(pixel_values)
            ocr_embeddings = ocr_feature.last_hidden_state

            with accelerator.accumulate(unet):
                # Convert images to latent space using frozen VAE
                latents = vae.encode(
                    batch["pixel_values"].to(weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Generate random noise for diffusion
                noise = torch.randn_like(latents)

                # Prepare mask and masked latents
                width, height, *_ = batch["masks"].size()[::-1]
                mask = torch.nn.functional.interpolate(
                    batch["masks"],
                    size=[width // vae_scale_factor, height // vae_scale_factor, *_][
                        :-2
                    ][::-1],
                )
                mask = mask.to(weight_dtype)

                # Get masked image latents
                masked_image_latents = vae.encode(
                    batch["masked_images"].to(weight_dtype)
                ).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                # Sample timesteps and add noise
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings for conditioning
                encoder_hidden_states = ocr_embeddings.detach()

                # Determine target based on prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Combine inputs for UNet
                model_input_latents = torch.cat(
                    [noisy_latents, mask, masked_image_latents], dim=1
                )

                # Get model prediction
                model_pred = unet(
                    model_input_latents, timesteps, encoder_hidden_states
                ).sample

                # Calculate loss and update model
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Optimization step
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
