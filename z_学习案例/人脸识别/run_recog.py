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
import argparse

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def collate_fn(x):
    return x[0]


def load_and_embed_images(image_folder, device):
    """
    加载并嵌入指定文件夹中的所有图片。

    :param image_folder: 图片所在的文件夹路径
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :return: 嵌入向量列表和对应的名称列表
    """
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
    dataset = datasets.ImageFolder(image_folder)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)

    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print("检测到的人脸及其概率: {:8f}".format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    # 图像嵌入
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    return embeddings, names


def calculate_distance(new_image_path, embeddings, names, device):
    """
    计算新图片与已有图片的距离。

    :param new_image_path: 新图片的路径
    :param embeddings: 已有图片的嵌入向量
    :param names: 已有图片的名称
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :return: 距离矩阵
    """
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

    # 加载新图片并进行人脸检测
    try:
        img = Image.open(new_image_path)  # 使用 PIL 加载图片
    except Exception as e:
        print(f"无法加载图片 {new_image_path}: {e}")
        return None

    # 使用 MTCNN 检测人脸
    x_aligned, prob = mtcnn(img, return_prob=True)
    if x_aligned is None:
        print(f"未检测到图片 {new_image_path} 中的人脸。")
        return None

    print(f"检测到的人脸及其概率: {prob:8f}")

    # 新图片的嵌入
    new_aligned = x_aligned.unsqueeze(0).to(device)  # 添加批次维度
    new_embedding = resnet(new_aligned).detach().cpu()

    # 计算新图片与已有图片的距离
    dists = [(new_embedding - e).norm().item() for e in embeddings]
    distance_df = pd.DataFrame([dists], columns=names, index=["New Image"])

    return distance_df


def main(
    image_folder, new_image_path, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    主函数，加载已有图片并计算新图片与它们的距离。

    :param image_folder: 已有图片所在的文件夹路径
    :param new_image_path: 新图片的路径
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :return: 距离矩阵
    """
    embeddings, names = load_and_embed_images(image_folder, device)
    logger.info(colored(f"names:{names}", "green"))
    distance_df = calculate_distance(new_image_path, embeddings, names, device)

    if distance_df is not None:
        print(distance_df)
    else:
        print("未检测到新图片中的人脸。")


if __name__ == "__main__":
    """
    python z_学习案例/人脸识别/run_recog.py z_using_files/img/p_face z_using_files/img/p_face/name5/image_5.png
    """

    parser = argparse.ArgumentParser(description="人脸识别脚本")
    parser.add_argument("image_folder", type=str, help="已有图片所在的文件夹路径")
    parser.add_argument("new_image_path", type=str, help="新图片的路径")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用的设备（'cuda' 或 'cpu'）",
    )

    args = parser.parse_args()

    main(args.image_folder, args.new_image_path, args.device)
