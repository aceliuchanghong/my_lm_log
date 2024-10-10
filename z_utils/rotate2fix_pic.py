import os
import cv2
from rapid_orientation import RapidOrientation
import argparse
import numpy as np
import random


def detect_text_orientation(image_path, output_dir="./upload_files/rotate_pics"):
    # 打开图像并将其转换为灰度图像
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    orientation_engine = RapidOrientation()
    orientation_res, _ = orientation_engine(img)
    rotation = int(orientation_res)
    # 根据检测的旋转角度来调整图像
    if rotation == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 保存或显示旋转后的图像
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    random_integer1 = random.randint(10000, 200000)
    random_integer2 = random.randint(10, 2000)
    corrected_image_path = os.path.join(
        f"{output_dir}", f"{random_integer1}_{random_integer2}.jpg"
    )

    cv2.imwrite(corrected_image_path, img)
    file_name = os.path.basename(image_path)
    if os.path.exists(os.path.join(output_dir, f"{file_name}")):
        os.remove(os.path.join(output_dir, f"{file_name}"))
    os.rename(corrected_image_path, os.path.join(output_dir, f"{file_name}"))

    return os.path.join(output_dir, f"{file_name}")


if __name__ == "__main__":
    """
    python z_utils/rotate2fix_pic.py --image_path z_using_files/pics/11.jpg --output_dir ./upload_files/rotate_pics
    """
    parser = argparse.ArgumentParser(description="检测并修正图像的文本方向")
    parser.add_argument("--image_path", type=str, help="输入图像的路径")
    parser.add_argument("--output_dir", type=str, help="输出图像的路径")
    args = parser.parse_args()
    corrected_image = detect_text_orientation(args.image_path, args.output_dir)
    print(f"Corrected image saved as: {corrected_image}")
