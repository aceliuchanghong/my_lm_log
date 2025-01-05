"""
指定文件夹 `C:\\Users\\lawrence\\Videos\\rick` 下面有文件夹,文件夹里面有视频
将全部视频转化为wav格式,存放到 `C:\\Users\\lawrence\\Videos\\rick_wav` 目录,原始文件名不变
python函数实现
uv pip install moviepy
"""

import os
from moviepy import VideoFileClip


def convert_videos_to_wav(input_folder, output_folder):
    """
    将指定文件夹下的所有视频转换为wav格式
    :param input_folder: 包含视频的文件夹路径
    :param output_folder: 保存wav文件的输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 只处理视频文件
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                try:
                    # 构建完整文件路径
                    video_path = os.path.join(root, file)
                    # 生成输出文件名
                    wav_filename = os.path.splitext(file)[0] + ".wav"
                    wav_path = os.path.join(output_folder, wav_filename)

                    # 使用moviepy进行转换
                    video = VideoFileClip(video_path)
                    video.audio.write_audiofile(wav_path)

                    print(f"成功转换: {file} -> {wav_filename}")
                except Exception as e:
                    print(f"转换失败: {file}, 错误: {str(e)}")


if __name__ == "__main__":
    # uv run test/usua2/trans_video_to_wav.py
    input_folder = r"C:\\Users\\lawrence\\Videos\\rick"
    output_folder = r"C:\\Users\\lawrence\\Videos\\rick_wav"
    convert_videos_to_wav(input_folder, output_folder)
