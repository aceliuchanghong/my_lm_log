from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
from PIL import Image

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# python z_学习案例/人脸识别/test_recog.py
workers = 0 if os.name == "nt" else 4
logger.info(colored(f"workers:{workers}", "green"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(colored(f"在 {device} 设备上运行", "green"))


def collate_fn(x):
    return x[0]


mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# MTCNN人脸检测
dataset = datasets.ImageFolder("z_using_files/img/p_face")
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print("检测到的人脸及其概率: {:8f}".format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

# 图像嵌入-输出距离矩阵
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
new_pd = pd.DataFrame(dists, columns=names, index=names)
logger.info(colored(f"new_pd:\n{new_pd}", "green"))

# 处理图像
img = Image.open("z_using_files/img/p_face/name5/image_5.png")
img_cropped = mtcnn(img, save_path="no_git_oic/pics/00.png")
img_embedding = resnet(img_cropped.unsqueeze(0))
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))
logger.info(colored(f"img_probs shape{img_probs.shape}", "green"))
logger.info(colored(f"img_probs:{img_probs}", "green"))
