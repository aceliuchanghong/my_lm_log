import ffmpeg
import argparse
import os

# pip install ffmpeg-python


def convert_audio(input_path: str, output_path: str) -> None:
    """
    使用ffmpeg将音频从一种格式转换为另一种格式。

    参数:
    input_path (str): 输入音频文件的路径。
    output_path (str): 输出音频文件的路径。

    返回:
    None
    """
    ffmpeg.input(input_path).output(output_path).run()


def main():
    parser = argparse.ArgumentParser(description="转换音频文件的脚本")
    parser.add_argument("input_file", type=str, help="输入音频文件路径")
    parser.add_argument("out_file", type=str, nargs="?", help="输出音频文件路径")
    args = parser.parse_args()

    if not args.out_file:
        output_dir = os.path.dirname(args.input_file)  # 使用输入文件目录作为输出目录
        input_filename = os.path.basename(args.input_file)
        base_filename, _ = os.path.splitext(input_filename)
        args.out_file = os.path.join(
            output_dir, base_filename + ".wav"
        )  # 生成默认的输出文件名

    convert_audio(args.input_file, args.out_file)


if __name__ == "__main__":
    # python z_utils/audio_trans.py "C:\Users\lawrence\Music\Wednesday at 10-44 PM.m4a..mp3" "00.wav"
    # python z_utils/audio_trans.py "C:\Users\lawrence\Music\Wednesday at 10-44 PM.m4a..mp3"
    # python z_utils/audio_trans.py "D:\机器学习记录\make_video\AI-数学-随机变量与概率分布-03\Monday at 7-45 PM.m4a..mp3"
    main()
