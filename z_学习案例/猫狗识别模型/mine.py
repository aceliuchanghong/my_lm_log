import os
import torch
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.utils import data
from dotenv import load_dotenv
import logging
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info(colored(f"{os.getcwd()}", "green"))

from torchvision import utils
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
from torch import optim
import time
from tempfile import TemporaryDirectory
from pathlib import *
from torchvision import models


def imageshow(img, title=None):
    img = img.numpy().transpose(
        (1, 2, 0)
    )  # tensor(color,width,heigth) →numpy (width,heigth,color)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    # Refresh the figure(刷新画面)
    plt.pause(1e-3)


imageshow(grid, title="Dogs vs Cats")
