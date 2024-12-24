import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pandas as pd
import argparse
import os
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


def load_and_embed_images(image_folder, device, mtcnn, resnet):
    """
    加载并嵌入指定文件夹中的所有图片。

    :param image_folder: 图片所在的文件夹路径
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :param mtcnn: MTCNN 模型
    :param resnet: InceptionResnetV1 模型
    :return: 嵌入向量列表和对应的名称列表
    """
    # MTCNN人脸检测
    dataset = datasets.ImageFolder(image_folder)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0], num_workers=0)

    aligned = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            # print("检测到的人脸及其概率: {:8f}".format(prob))
            aligned.append(x_aligned)

    # 图像嵌入
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    return embeddings


def calculate_distance(new_image_path, embeddings, device, mtcnn, resnet):
    """
    计算新图片与已有图片的距离。

    :param new_image_path: 新图片的路径
    :param embeddings: 已有图片的嵌入向量
    :param names: 已有图片的名称
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :param mtcnn: MTCNN 模型
    :param resnet: InceptionResnetV1 模型
    :return: 距离矩阵
    """
    # 加载新图片并进行人脸检测
    try:
        img = Image.open(new_image_path)  # 使用 PIL 加载图片
    except Exception as e:
        print(f"无法加载图片 {new_image_path}: {e}")
        return None

    # 使用 MTCNN 检测人脸
    x_aligned, prob = mtcnn(img, return_prob=True)
    if x_aligned is None:
        logger.info(colored(f"未检测到图片 {new_image_path} 中的人脸", "red"))
        return None

    # print(f"检测到的人脸及其概率: {prob:8f}")

    # 新图片的嵌入
    new_aligned = x_aligned.unsqueeze(0).to(device)  # 添加批次维度
    new_embedding = resnet(new_aligned).detach().cpu()

    # 计算新图片与已有图片的距离
    dists = [(new_embedding - e).norm().item() for e in embeddings]
    distance_df = pd.DataFrame([dists], index=["New Image"])

    return distance_df


def main(
    image_folder,
    new_image_path,
    device="cuda:3" if torch.cuda.is_available() else "cpu",
    model_choose_num=1,
):
    """
    主函数，加载已有图片并计算新图片与它们的距离。

    :param image_folder: 已有图片所在的文件夹路径
    :param new_image_path: 新图片的路径
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :param model_choose_num: 选择的模型编号
    :return: 距离矩阵
    """
    model_detail = {
        "no_git_oic/20180408-102900-casia-webface.pt": 10575,
        "no_git_oic/20180402-114759-vggface2.pt": 8631,
    }
    model_name = [
        "no_git_oic/20180408-102900-casia-webface.pt",
        "no_git_oic/20180402-114759-vggface2.pt",
    ]

    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
    )
    resnet = (
        InceptionResnetV1(
            pretrained=model_name[model_choose_num],
            tmp_classes=model_detail[model_name[model_choose_num]],
        )
        .eval()
        .to(device)
    )

    embeddings = load_and_embed_images(image_folder, device, mtcnn, resnet)
    distance_df = calculate_distance(new_image_path, embeddings, device, mtcnn, resnet)

    if distance_df is not None:
        logger.info(colored(f"\n{distance_df}", "green"))
    else:
        print("未检测到新图片中的人脸。")


if __name__ == "__main__":
    """
    源库修改
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None, tmp_classes=8631):

    def load_weights(mdl, name):
        if name == "vggface2":
            path = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
        elif name == "casia-webface":
            path = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt"
        elif isfile(name):
            path = name
        else:
            raise ValueError(
                'Pretrained models only exist for "vggface2" and "casia-webface", or provide a valid local file path'
            )

        model_dir = os.path.join(get_torch_home(), "checkpoints")
        os.makedirs(model_dir, exist_ok=True)

        if isfile(path):
            cached_file = path
        else:
            cached_file = os.path.join(model_dir, os.path.basename(path))
            if not os.path.exists(cached_file):
                download_url_to_file(path, cached_file)

        state_dict = torch.load(cached_file)
        mdl.load_state_dict(state_dict)
    """
    parser = argparse.ArgumentParser(description="人脸识别脚本")
    parser.add_argument("image_folder", type=str, help="已有图片所在的文件夹路径")
    parser.add_argument("new_image_path", type=str, help="新图片的路径")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3" if torch.cuda.is_available() else "cpu",
        help="使用的设备（'cuda' 或 'cpu'）",
    )
    parser.add_argument(
        "--model_choose_num",
        type=int,
        choices=[0, 1],
        default=1,
        help="选择的模型0-casia-webface,1-vggface2",
    )

    args = parser.parse_args()
    """
    python z_学习案例/人脸识别/run_recog.py z_using_files/img/p_face z_using_files/img/p_face/name5/image_5.png
    --model_choose_num 0
    """

    main(args.image_folder, args.new_image_path, args.device, args.model_choose_num)
